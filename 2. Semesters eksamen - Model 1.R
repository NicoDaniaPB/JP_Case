pacman::p_load(tidyverse, lubridate, caret, randomForest, xgboost, pROC)

# 1. Indlæsning af data -------------------------------------------------------
model_data_raw <- read_rds("data/renset_datasæt.rds")

# ➤ NYT: Gem pseudo_id i model_data_raw (vi fjerner det ikke endnu)
# pseudo_id skal følge ALLE transformationer
# så vi lader det blive i model_data indtil lige før modellering

# 2. Yderligere klargøring af data til modellering ------------------------

# Filtrering til kunder med target 
model_data <- model_data_raw %>%
  filter(!is.na(continued_after_campaign))

# Datoformatering
model_data <- model_data %>%
  mutate(
    order_date = as.Date(order_date),
    usr_created = as.Date(usr_created),
    subscription_cancel_date = as.Date(subscription_cancel_date),
    birthdate = as.Date(birthdate)
  )

# ➤ RETTET: Fjern kun tekststrenge – pseudo_id beholdes
model_data <- model_data %>%
  select(-order_trackertag, -reason)

# Target som factor
model_data <- model_data %>%
  mutate(
    continued_after_campaign = factor(
      continued_after_campaign,
      levels = c(0, 1),
      labels = c("No", "Yes")
    )
  )

# 3. Feature engineering -----------------------------------------------------

model_data <- model_data %>%
  mutate(
    days_active_before_campaign = account_active_days,
    days_since_user_created = as.numeric(order_date - usr_created),
    days_between_signup_and_order = as.numeric(order_date - usr_created),
    days_to_cancel = as.numeric(subscription_cancel_date - order_date),
    age_at_order = as.numeric(difftime(order_date, birthdate, units = "days")) / 365,
    permission_user = ifelse(permission_given_order == TRUE, 1, 0),
    total_previous_engagement = previous_subscriptions + previous_campaigns + previous_trials,
    has_previous_subscriptions = ifelse(previous_subscriptions > 0, 1, 0),
    engagement_score = (visits * 0.4) + (unique_pages * 0.3) + (avg_scroll * 0.3),
    mobile_heavy = ifelse(mobile_ratio > 0.7, 1, 0),
    age_engagement_interaction = age * engagement_score,
    permission_engagement = permission_user * engagement_score
  ) %>%
  drop_na()

# 4. Leakage-variabler ----------------------------------------------------

model_data <- model_data %>%
  select(
    -subscription_cancel_date,
    -days_to_cancel,
    -end_date,
    -expiration_date,
    -newsletters_after_order,
    -fast_churn,
    -subscription_length_days,
    -length_group,
    -type,
    -order_date,
    -usr_created,
    -birthdate,
    -first_campaign_day,
    -last_campaign_day,
    -days_between_signup_and_order
  )

# ➤ NYT: Fjern pseudo_id lige før modellering
pseudo_ids <- model_data$pseudo_id
model_data <- model_data %>% select(-pseudo_id)

# 5. Train/test split --------------------------------------------------------

set.seed(47)

train_index <- createDataPartition(model_data$continued_after_campaign, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

# ➤ NYT: Split pseudo_ids korrekt
test_ids <- pseudo_ids[-train_index]

# 6. Logistisk regression ----------------------------------------------------

set.seed(47)

cv_ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

log_cv <- train(
  continued_after_campaign ~ .,
  data = train_data,
  method = "glm",
  family = binomial,
  trControl = cv_ctrl,
  metric = "ROC"
)

log_model <- glm(
  continued_after_campaign ~ ., 
  data = train_data,
  family = binomial
)

log_prob <- predict(log_model, test_data, type = "response")

roc_log <- roc(test_data$continued_after_campaign, log_prob)
cut_log <- as.numeric(coords(roc_log, "best", ret = "threshold"))

log_pred <- ifelse(log_prob > cut_log, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))

cm_log <- confusionMatrix(log_pred, test_data$continued_after_campaign)

# 7. Random Forest -----------------------------------------------------------

set.seed(47)

rf_cv <- train(
  continued_after_campaign ~ .,
  data = train_data,
  method = "rf",
  trControl = cv_ctrl,
  metric = "ROC",
  tuneLength = 5
)

rf_model <- randomForest(
  continued_after_campaign ~ .,
  data = train_data,
  ntree = 500,
  mtry = floor(sqrt(ncol(train_data) - 1)),
  importance = TRUE
)

rf_prob <- predict(rf_model, test_data, type = "prob")[,2]

roc_rf <- roc(test_data$continued_after_campaign, rf_prob)
cut_rf <- as.numeric(coords(roc_rf, "best", ret = "threshold"))

rf_pred <- ifelse(rf_prob > cut_rf, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))

cm_rf <- confusionMatrix(rf_pred, test_data$continued_after_campaign)

# 8. XGBoost-----------------------------------------------------------------

full_matrix <- model.matrix(continued_after_campaign ~ . - 1, data = model_data)

train_matrix <- full_matrix[train_index, ]
test_matrix  <- full_matrix[-train_index, ]

y_train <- model_data$continued_after_campaign[train_index]

dtrain <- xgb.DMatrix(data = train_matrix, label = as.numeric(y_train) - 1)
dtest  <- xgb.DMatrix(data = test_matrix,  label = as.numeric(model_data$continued_after_campaign[-train_index]) - 1)

set.seed(47)

xgb_cv2 <- xgb.cv(
  data = dtrain,
  nrounds = 300,
  params = list(
    max_depth = 4,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = "binary:logistic",
    eval_metric = "auc"
  ),
  nfold = 5,
  verbose = 0
)

xgb_model <- xgb.train(
  data = dtrain,
  nrounds = 300,
  params = list(
    max_depth = 4,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = "binary:logistic",
    eval_metric = "auc"
  )
)

xgb_prob <- predict(xgb_model, dtest)

roc_xgb <- roc(test_data$continued_after_campaign, xgb_prob)
cut_xgb <- as.numeric(coords(roc_xgb, "best", ret = "threshold"))

xgb_pred <- ifelse(xgb_prob > cut_xgb, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))

cm_xgb <- confusionMatrix(xgb_pred, test_data$continued_after_campaign)

# 9. Tabel over mest risikable kunder -------------------------------------

risk_list <- tibble(
  pseudo_id = test_ids,
  churn_probability = xgb_prob,
  predicted_class = xgb_pred
)

risk_list_sorted <- risk_list %>%
  arrange(desc(churn_probability))

head(risk_list_sorted, 50)

risk_table <- risk_list_sorted %>%
  mutate(
    churn_risk_pct = round(churn_probability * 100, 1),
    risk_group = case_when(
      churn_probability >= 0.75 ~ "Høj risiko",
      churn_probability >= 0.50 ~ "Middel risiko",
      TRUE ~ "Lav risiko"
    )
  ) %>%
  select(
    pseudo_id,
    churn_risk_pct,
    predicted_class,
    risk_group
  )

head(risk_table, 50)
view(risk_table)

# 10. Økonomisk analyse af churn-kampagne -------------------------------------

# Listepris efter kampagne 
JP_pris <- 199  

# Vi sætter et forventet retention uplift til 15% 
retention_uplift_15percent <- 0.15  

# Definition af højrisiko-kunder
high_risk <- risk_list_sorted %>% 
  filter(churn_probability > 0.75)

n_high_risk <- nrow(high_risk)

# Forventet antal reddede kunder
saved_customers_15percent <- n_high_risk * retention_uplift_15percent

# Scenarier for 3, 6 og 12 måneder
value_3m_15percent  <- saved_customers_15percent * JP_pris * 3
value_6m_15percent  <- saved_customers_15percent * JP_pris * 6
value_12m_15percent <- saved_customers_15percent * JP_pris * 12

# Samlet økonomitabel
economy_table_15percent <- tibble(
  periode = c("3 måneder", "6 måneder", "12 måneder"),
  højrisiko_kunder = n_high_risk,
  reddede_kunder = round(saved_customers_15percent),
  besparelse_kr = round(c(value_3m_15percent, value_6m_15percent, value_12m_15percent))
)

economy_table_15percent

# Vi laver også en med 5% (worst case scenario):
retention_uplift_5percent <- 0.05  

# Forventet antal reddede kunder
saved_customers_5percent <- n_high_risk * retention_uplift_5percent

# Scenarier for 3, 6 og 12 måneder
value_3m_5percent  <- saved_customers_5percent * JP_pris * 3
value_6m_5percent  <- saved_customers_5percent * JP_pris * 6
value_12m_5percent <- saved_customers_5percent * JP_pris * 12

# Samlet økonomitabel
economy_table_5percent <- tibble(
  periode = c("3 måneder", "6 måneder", "12 måneder"),
  højrisiko_kunder = n_high_risk,
  reddede_kunder = round(saved_customers_5percent),
  besparelse_kr = round(c(value_3m_5percent, value_6m_5percent, value_12m_5percent))
)

economy_table_5percent

# Vi laver også en med 25% (meget optimistisk):
retention_uplift_25percent <- 0.25  

# Forventet antal reddede kunder
saved_customers_25percent <- n_high_risk * retention_uplift_25percent

# Scenarier for 3, 6 og 12 måneder
value_3m_25percent  <- saved_customers_25percent * JP_pris * 3
value_6m_25percent  <- saved_customers_25percent * JP_pris * 6
value_12m_25percent <- saved_customers_25percent * JP_pris * 12

# Samlet økonomitabel
economy_table_25percent <- tibble(
  periode = c("3 måneder", "6 måneder", "12 måneder"),
  højrisiko_kunder = n_high_risk,
  reddede_kunder = round(saved_customers_25percent),
  besparelse_kr = round(c(value_3m_25percent, value_6m_25percent, value_12m_25percent))
)

economy_table_25percent




# 11. AUC-sammenligning -------------------------------------------------------

auc_log <- as.numeric(pROC::auc(roc_log))
auc_rf  <- as.numeric(pROC::auc(roc_rf))
auc_xgb <- as.numeric(pROC::auc(roc_xgb))

model_sammenligning <- tibble(
  Model       = c("Logistisk regression", "Random Forest", "XGBoost"),
  AUC         = round(c(auc_log, auc_rf, auc_xgb), 4),
  Accuracy    = round(c(
    cm_log$overall["Accuracy"],
    cm_rf$overall["Accuracy"],
    cm_xgb$overall["Accuracy"]
  ), 4),
  Sensitivity = round(c(
    cm_log$byClass["Sensitivity"],
    cm_rf$byClass["Sensitivity"],
    cm_xgb$byClass["Sensitivity"]
  ), 4),
  Specificity = round(c(
    cm_log$byClass["Specificity"],
    cm_rf$byClass["Specificity"],
    cm_xgb$byClass["Specificity"]
  ), 4)
)

print(model_sammenligning)

plot(roc_log, col = "steelblue", lwd = 2,
     main = "ROC-kurver — Model 1 (churn ved kampagneslut)")
plot(roc_rf,  col = "darkgreen", lwd = 2, add = TRUE)
plot(roc_xgb, col = "firebrick", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(
         paste0("Logistisk regression (AUC = ", round(auc_log, 3), ")"),
         paste0("Random Forest        (AUC = ", round(auc_rf,  3), ")"),
         paste0("XGBoost              (AUC = ", round(auc_xgb, 3), ")")
       ),
       col = c("steelblue", "darkgreen", "firebrick"),
       lwd = 2)

# 12. Variabelvigtighed -------------------------------------------------------

# Random Forest: built-in importance
# MeanDecreaseGini måler hvor meget hver variabel bidrager til
# at reducere usikkerhed på tværs af alle træer i skoven
rf_imp <- importance(rf_model) %>%
  as.data.frame() %>%
  rownames_to_column("variabel") %>%
  arrange(desc(MeanDecreaseGini))

# Vi plotter de 15 vigtigste variabler
rf_imp %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(variabel, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "darkgreen", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Variabelvigtighed — Random Forest (Model 1)",
    subtitle = "MeanDecreaseGini: hvor meget variablen reducerer usikkerhed på tværs af træer",
    x = NULL,
    y = "MeanDecreaseGini"
  ) +
  theme_minimal()

# XGBoost: feature importance
xgb_imp <- xgb.importance(model = xgb_model) %>%
  as_tibble()

# Vi plotter de 15 vigtigste variabler
xgb_imp %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "firebrick", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Variabelvigtighed — XGBoost (Model 1)",
    subtitle = "Gain: hvor meget hver variabel forbedrer modellens præcision",
    x = NULL,
    y = "Gain"
  ) +
  theme_minimal()


# 13. Cross validation sammenligning --------------------------------------

# Udtræk af CV-resultater fra logistisk regression (caret-pakken)
log_cv_results <- log_cv$results %>%
  select(ROC, Sens, Spec) %>%
  slice(which.max(ROC)) %>%
  mutate(Model = "Logistisk regression")

# Udtræk af CV-resultater fra Random Forest (caret-pakken)
rf_cv_results <- rf_cv$results %>%
  select(ROC, Sens, Spec) %>%
  slice(which.max(ROC)) %>%
  mutate(Model = "Random Forest")

# Udtræk  af CV-resultater fra XGBoost (xgb.cv)
xgb_cv_results <- tibble(
  Model = "XGBoost",
  ROC   = max(xgb_cv2$evaluation_log$test_auc_mean),
  Sens  = NA,   # xgb.cv giver ikke Sens/Spec direkte
  Spec  = NA
)

# Vi samler alle CV-resultater i en tabel
cv_sammenligning <- bind_rows(
  log_cv_results,
  rf_cv_results,
  xgb_cv_results
) %>%
  select(Model, ROC, Sens, Spec)

# Vi ser resultatet
print(cv_sammenligning)

# Vi kan se, at XGboost modellen opnår den højeste cross validation ROC på 
# 89,45%. Hvilket betyder, at det er den der generaliser bedst på nye ukendte 
# data. 

# 14. Konklusion til ML-model nr. 1 --------------------------------------------

# Vi har udviklet en klassifikationsmodel, der forudsiger, om en kunde 
# fortsætter efter kampagneperioden. Vi har brugt korrekt feature engineering og
# fjernet dataleakage, og lavet tre forskellige klassifikationsmodeller. XGBoost
# modellen opnår den højeste AUC på 0.931 og den bedste balance mellem 
# sensitivity og specificity. Det er derfor den bedste model til at forudsige 
# churn efter kampagnen. 
# XGBoost-model har en accuracy på 86,3%. Modellen identificerer 88% af 
# churnerne og 85% af de kunder, der fortsætter, hvilket gør den velegnet til 
# at understøtte JPs problemstillinger. 

# 15. Gem som RDS-filer til Shiny App ------------------------------------------

saveRDS(xgb_model, "xgb_model.rds")





