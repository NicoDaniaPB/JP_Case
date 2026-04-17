pacman::p_load(tidyverse, lubridate, caret, pROC, randomForest, xgboost)

# Vi har nu loaded de pakker vi skal bruge. Vi bruger tidyverse til 
# datamanipulation. lubridate bruges til at håndterer datoer. caret bruges til 
# train/test split og modelværktøjer. pROC bruges til ROC-kurver og cutoffs i 
# vores klassifikationsmodeller. randomForest- og xgboost- pakkerne bruges til
# at køre RF og Boosting modellerne.

# Vores anden model skal kunne klassificere, om kunder der fortsætter efter
# kampagneperioden churner inden for 30 dage eller ej.

# 1. Indlæsning af data ---------------------------------------
model_data <- read_rds("data/renset_datasæt.rds")

# 2. Yderligere klargøring af data til modellerne -------------------------

# Filtrering til kunder der FORTSÆTTER efter kampagnen
model2_data <- model_data %>%
  filter(continued_after_campaign == 1)

# Target-variabel: churn_30
# Vi konverterer til factor med gyldige labels til caret
model2_data <- model2_data %>%
  mutate(
    churn_30 = factor(
      churn_30,
      levels = c(0, 1),
      labels = c("No", "Yes")
    )
  )

# 3. Feature engineering --------------------------------------------------

# Vi opretter nu nogle nye features, som skal være med til at øge modellens 
# forklaringskraft. 
model2_data <- model2_data %>%
  mutate(
    days_since_user_created = as.numeric(difftime(order_date, usr_created, units = "days")),
    age_at_order = as.numeric(difftime(order_date, birthdate, units = "days")) / 365,
    total_previous_engagement = previous_subscriptions + previous_campaigns + previous_trials,
    has_previous_subscriptions = ifelse(previous_subscriptions > 0, 1, 0)
  )

# 4. Udvælgelse af variabler ------------------------------------------
model2_data <- model2_data %>%
  select(
    # Demografiske
    koen,
    age,
    
    # Abonnementshistorik
    previous_subscriptions,
    previous_campaigns,
    previous_trials,
    account_active_days,
    
    # Permission og nyhedsbreve
    permission_given_order,
    newsletters_before_order,
    newsletters_after_order,
    
    # Adfærd de første 30 dage af kampagnen
    visits,
    unique_pages,
    restricted_views,
    restricted_ratio,
    avg_scroll,
    mobile_ratio,
    desktop_ratio,
    
    # Feature engineered
    days_since_user_created,
    age_at_order,
    total_previous_engagement,
    has_previous_subscriptions,
    
    # Target
    churn_30
  ) %>%
  drop_na()

# Vi har kun beholdt variabler som er kendte ved eller kort efter køb.
# permission_given_today er fjernet da den er målt pr. 10. marts 2026 (fremtidig info).
# Adfærdsvariablerne er målt de første 30 dage efter køb, hvilket er inden
# kampagnens udløb, og giver derfor ikke dataleakage.

# 5. Train/test split -----------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi laver et 80/20 split 
train_index2 <- createDataPartition(model2_data$churn_30, p = 0.8, list = FALSE)
train_data2 <- model2_data[train_index2, ]
test_data2  <- model2_data[-train_index2, ]

# 6. Logistisk regression -------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Cross validation
cv_ctrl2 <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

log_cv2 <- train(
  churn_30 ~ .,
  data = train_data2,
  method = "glm",
  family = binomial,
  trControl = cv_ctrl2,
  metric = "ROC"
)

log_cv2

# Vi træner en logistisk regressionsmodel på alle vores variabler
log_model2 <- glm(churn_30 ~ ., data = train_data2, family = binomial)

# Prediction - Vi forudsiger sandsynligheder for vores testdatasæt
log_prob2 <- predict(log_model2, test_data2, type = "response")

# Vi bereger ROC-kurven og finder det optimale cutoff
roc_log2 <- roc(test_data2$churn_30, log_prob2)
cut_log2 <- as.numeric(coords(roc_log2, "best", ret = "threshold"))

# Vi konverterer sandsynlighederne til klasser og beregner confusion matrix
log_pred2 <- ifelse(log_prob2 > cut_log2, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))
cm_log2 <- confusionMatrix(log_pred2, test_data2$churn_30)
cm_log2

# 7. Random Forest --------------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Cross validation
rf_cv2 <- train(
  churn_30 ~ .,
  data = train_data2,
  method = "rf",
  trControl = cv_ctrl2,
  metric = "ROC",
  tuneLength = 5
)

rf_cv2

# Vi laver vores Random Forest model
rf_model2 <- randomForest(
  churn_30 ~ .,
  data = train_data2,
  ntree = 500,
  mtry = floor(sqrt(ncol(train_data2) - 1)),
  importance = TRUE
)

# Prediction - Vi forudsiger sandsynligheder for vores testdatasæt
rf_prob2 <- predict(rf_model2, test_data2, type = "prob")[,2]

# Vi bereger ROC-kurven og finder det optimale cutoff
roc_rf2 <- roc(test_data2$churn_30, rf_prob2)
cut_rf2 <- as.numeric(coords(roc_rf2, "best", ret = "threshold"))

# Klassifikation
rf_pred2 <- ifelse(rf_prob2 > cut_rf2, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))

# Vi laver vores Confusion Matrix
cm_rf2 <- confusionMatrix(rf_pred2, test_data2$churn_30)
cm_rf2

# 8. XGBoost ---------------------------------------------------

# Vi laver en samlet model.matrix for hele datasættet
full_matrix2 <- model.matrix(churn_30 ~ . - 1, data = model2_data)

# Vi konverterer target til numerisk
full_label2  <- as.numeric(model2_data$churn_30) - 1

# Vi splitter matrix og labels i train/test
train_matrix2 <- full_matrix2[train_index2, ]
test_matrix2  <- full_matrix2[-train_index2, ]

# Brug den originale factor-target til CV
y_train2 <- model2_data$churn_30[train_index2]

train_label2 <- full_label2[train_index2]
test_label2  <- full_label2[-train_index2]

# Vi laver en DMatrix
dtrain2 <- xgb.DMatrix(data = train_matrix2, label = as.numeric(y_train2) - 1)
dtest2  <- xgb.DMatrix(data = test_matrix2,
                       label = as.numeric(model2_data$churn_30[-train_index2]) - 1)

# Vi sikrer reproducerbarhed
set.seed(47)

# Cross validation
xgb_cv2 <- xgb.cv(
  data = dtrain2,
  nrounds = 300,
  max_depth = 4,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  objective = "binary:logistic",
  eval_metric = "auc",
  nfold = 5,
  verbose = 0
)

xgb_cv2

# Vi træner vores XGBoost-model
xgb_model2 <- xgb.train(
  params = list(
    max_depth = 4,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = "binary:logistic",
    eval_metric = "auc"
  ),
  data = dtrain2,
  nrounds = 300
)

# Prediction 
xgb_prob2 <- predict(xgb_model2, dtest2)

# ROC og cutoff
roc_xgb2 <- roc(test_data2$churn_30, xgb_prob2)
cut_xgb2 <- as.numeric(coords(roc_xgb2, "best", ret = "threshold"))

# Klassifikation
xgb_pred2 <- ifelse(xgb_prob2 > cut_xgb2, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))

# Vi laver vores Confusion Matrix
cm_xgb2 <- confusionMatrix(xgb_pred2, test_data2$churn_30)
cm_xgb2

# 9. AUC-sammenligning -------------------------------------------------------

auc_log2 <- as.numeric(pROC::auc(roc_log2))
auc_rf2  <- as.numeric(pROC::auc(roc_rf2))
auc_xgb2 <- as.numeric(pROC::auc(roc_xgb2))

model_sammenligning2 <- tibble(
  Model       = c("Logistisk regression", "Random Forest", "XGBoost"),
  AUC         = round(c(auc_log2, auc_rf2, auc_xgb2), 4),
  Accuracy    = round(c(
    cm_log2$overall["Accuracy"],
    cm_rf2$overall["Accuracy"],
    cm_xgb2$overall["Accuracy"]
  ), 4),
  Sensitivity = round(c(
    cm_log2$byClass["Sensitivity"],
    cm_rf2$byClass["Sensitivity"],
    cm_xgb2$byClass["Sensitivity"]
  ), 4),
  Specificity = round(c(
    cm_log2$byClass["Specificity"],
    cm_rf2$byClass["Specificity"],
    cm_xgb2$byClass["Specificity"]
  ), 4)
)

print(model_sammenligning2)

plot(roc_log2, col = "steelblue", lwd = 2,
     main = "ROC-kurver — Model 2 (churn_30 efter kampagneslut)")
plot(roc_rf2,  col = "darkgreen", lwd = 2, add = TRUE)
plot(roc_xgb2, col = "firebrick", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(
         paste0("Logistisk regression (AUC = ", round(auc_log2, 3), ")"),
         paste0("Random Forest        (AUC = ", round(auc_rf2,  3), ")"),
         paste0("XGBoost              (AUC = ", round(auc_xgb2, 3), ")")
       ),
       col = c("steelblue", "darkgreen", "firebrick"),
       lwd = 2)

# 10. Variabelvigtighed -------------------------------------------------------

# Random Forest: built-in importance
# MeanDecreaseGini måler hvor meget hver variabel bidrager til
# at reducere usikkerhed på tværs af alle træer i skoven
rf_imp2 <- importance(rf_model2) %>%
  as.data.frame() %>%
  rownames_to_column("variabel") %>%
  arrange(desc(MeanDecreaseGini))

# Vi plotter de 15 vigtigste variabler
rf_imp2 %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(variabel, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "darkgreen", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Variabelvigtighed — Random Forest (Model 2)",
    subtitle = "MeanDecreaseGini: hvor meget variablen reducerer usikkerhed på tværs af træer",
    x = NULL,
    y = "MeanDecreaseGini"
  ) +
  theme_minimal()

# XGBoost: feature importance
xgb_imp2 <- xgb.importance(model = xgb_model2) %>%
  as_tibble()

# Vi plotter de 15 vigtigste variabler
xgb_imp2 %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "firebrick", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Variabelvigtighed — XGBoost (Model 2)",
    subtitle = "Gain: hvor meget hver variabel forbedrer modellens præcision",
    x = NULL,
    y = "Gain"
  ) +
  theme_minimal()

# 11. Cross validation sammenligning -----------------------------------------

# Udtræk af CV-resultater fra logistisk regression (caret-pakken)
log_cv2_results <- log_cv2$results %>%
  select(ROC, Sens, Spec) %>%
  slice(which.max(ROC)) %>%
  mutate(Model = "Logistisk regression")

# Udtræk af CV-resultater fra Random Forest (caret-pakken)
rf_cv2_results <- rf_cv2$results %>%
  select(ROC, Sens, Spec) %>%
  slice(which.max(ROC)) %>%
  mutate(Model = "Random Forest")

# Udtræk af CV-resultater fra XGBoost (xgb.cv)
xgb_cv2_results <- tibble(
  Model = "XGBoost",
  ROC   = max(xgb_cv2$evaluation_log$test_auc_mean),
  Sens  = NA,   # xgb.cv giver ikke Sens/Spec direkte
  Spec  = NA
)

# Vi samler alle CV-resultater i en tabel
cv_sammenligning2 <- bind_rows(
  log_cv2_results,
  rf_cv2_results,
  xgb_cv2_results
) %>%
  select(Model, ROC, Sens, Spec)

print(cv_sammenligning2)

# 12. Konklusion til model nr. 2 -------------------------------------------

# Vi har udviklet en klassifikationsmodel til at forudsige churn_30 blandt
# kunder der fortsætter efter kampagneperioden. XGBoost er den bedste model
# med AUC = 0.788 og en god balance mellem sensitivity (85.7%) og
# specificity (59.4%), hvilket betyder modellen er god til at identificere
# kunder der churner inden for 30 dage efter kampagnens afslutning.
# Random Forest har lavere AUC (0.757) og dårligere specificity (56.2%),
# hvilket betyder den misser flere churn-kandidater end XGBoost.
# Logistisk regression er ikke egnet med AUC på 0.650.
# Den vigtigste forklarende variabel er account_active_days — kunder med
# lang historik hos JP churner markant sjældnere end nye kunder.

# 13. Gem som RDS-fil til Shiny App ------------------------------------------


