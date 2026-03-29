pacman::p_load(tidyverse, lubridate, caret, randomForest, xgboost, pROC)

# Vi har nu loaded de pakker vi skal bruge. Vi bruger tidyverse til 
# datamanipulation. lubridate bruges til at håndterer datoer. caret bruges til 
# train/test split og modelværktøjer. pROC bruges til ROC-kurver og cutoffs i 
# vores klassifikationsmodeller. randomForest- og xgboost- pakkerne bruges til
# at køre RF og boosting modellerne. 

# Vores første model skal kunne klassificere, om kunden forsætter efter
# kampagneperioden, eller om de churner. 

# 1. Indlæsning af data -------------------------------------------------------
model_data <- read_rds("data/renset_datasæt.rds")

# 2. Yderligere klargøring af data til modellering ------------------------

# Filtrering til kunder med target 
model_data <- model_data %>%
  filter(!is.na(continued_after_campaign))
# Her har vi fjernet alle rækkere, hvor vores mål-variabel mangler.

# Vi opretter en ny variabel "campaign_group", som skal bruges som en 
# forklarende variabel til vores klassifikationsmodeller. Vi sørger for, at 
# variablen er en factor. 
model_data <- model_data %>%
  mutate(
    campaign_group = case_when(
      str_detect(order_trackertag, "Basis") ~ "Basis",
      str_detect(order_trackertag, "Marts") ~ "Marts",
      str_detect(order_trackertag, "Biler") ~ "Biler",
      str_detect(order_trackertag, "FB") ~ "Facebook",
      str_detect(order_trackertag, "adwords") ~ "Google",
      TRUE ~ "Other"
    ) %>% as.factor()
  )

# Fjerning af ID'er og tekststrenge, som vi ikke skal bruge i modellen, da det
# vil give en fejlmeddelse. 
model_data <- model_data %>%
  select(-pseudo_id, -order_trackertag, -reason)

# 3. Feature engineering -----------------------------------------------------

# Vi opretter nu nogle nye features, som skal være med til at øge modellens 
# forklaringskraft. Udover det, så fjerner vi også NA-værdierne. 

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
    permission_engagement = permission_user * engagement_score,
    continued_after_campaign = as.factor(continued_after_campaign)
  ) %>%
  drop_na()

# 4. Leakage-variabler ----------------------------------------------------

# Vi fjerner nu alle leaker-variabler i modellen:
model_data <- model_data %>%
  select(
    -subscription_cancel_date,
    -days_to_cancel,
    -end_date,
    -expiration_date,
    -newsletters_after_order,
    -fast_churn,
    -subscription_length_days,
    -length_group
  )

# Vi har nu fjerner de variabler, som indeholder infromation om fremtiden
# (fx opsigelsesdato). Dette forhinder dataleakage, og sikrer at vi har en 
# realistisk model, som kan anvendes i praksis. Vi bruger select til at fjerne
# variablerne. 

# 5. Train/test split --------------------------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi laver et 80/20 split på vores datasæt
train_index <- createDataPartition(model_data$continued_after_campaign, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

# Vi ensretter factor-levels
train_data$campaign_group <- factor(train_data$campaign_group)
test_data$campaign_group  <- factor(test_data$campaign_group,
                                    levels = levels(train_data$campaign_group))

# 6. Logistisk regression ----------------------------------------------------

# Vi træner en logistisk regressionsmodel på alle vores variabler
log_model <- glm(
  continued_after_campaign ~ ., 
  data = train_data,
  family = binomial
)

# Prediction - Vi forudsiger sandsynligheder for vores testdatasæt
log_prob <- predict(log_model, test_data, type = "response")

# Vi bereger ROC-kurven og finder det optimale cutoff
roc_log <- roc(test_data$continued_after_campaign, log_prob)
cut_log <- as.numeric(coords(roc_log, "best", ret = "threshold"))

# Vi konverterer sandsynlighederne til klasser og beregner confusion matrix
log_pred <- ifelse(log_prob > cut_log, "1", "0") %>% factor(levels = c("0","1"))
cm_log <- confusionMatrix(log_pred, test_data$continued_after_campaign)
cm_log


# 7. Random Forest -----------------------------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi laver vores Random Forest model
rf_model <- randomForest(
  continued_after_campaign ~ .,
  data = train_data,
  ntree = 500,
  mtry = floor(sqrt(ncol(train_data) - 1)),
  importance = TRUE
)

# Prediction - Vi forudsiger sandsynligheder for vores testdatasæt
rf_prob <- predict(rf_model, test_data, type = "prob")[,2]

# Vi bereger ROC-kurven og finder det optimale cutoff
roc_rf <- roc(test_data$continued_after_campaign, rf_prob)
cut_rf <- as.numeric(coords(roc_rf, "best", ret = "threshold"))

# Klassifikation
rf_pred <- ifelse(rf_prob > cut_rf, "1", "0") %>% factor(levels = c("0","1"))

# # Vi laver vores Confusion Matrix
cm_rf <- confusionMatrix(rf_pred, test_data$continued_after_campaign)
cm_rf

# 8. XGBoost-----------------------------------------------------------------

# Vi laver en samlet model.matrix for hele datasættet
full_matrix <- model.matrix(continued_after_campaign ~ . - 1, data = model_data)

# Vi konverterer target til numerisk
full_label <- as.numeric(model_data$continued_after_campaign) - 1

# Vi splitter matrix og labels i train/test
train_matrix <- full_matrix[train_index, ]
test_matrix  <- full_matrix[-train_index, ]

train_label <- full_label[train_index]
test_label  <- full_label[-train_index]

# Vi lav en DMatrix
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix,  label = test_label)

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi træner vores XGBoost-model
xgb_model <- xgb.train(
  data = dtrain,
  nrounds = 300,
  max_depth = 4,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  objective = "binary:logistic",
  eval_metric = "auc"
)

# Prediction 
xgb_prob <- predict(xgb_model, dtest)

# ROC og cutoff
roc_xgb <- roc(test_data$continued_after_campaign, xgb_prob)
cut_xgb <- as.numeric(coords(roc_xgb, "best", ret = "threshold"))

# Klassifikation
xgb_pred <- ifelse(xgb_prob > cut_xgb, "1", "0") %>% factor(levels = c("0","1"))

# Vi laver vores Confusion Matrix
cm_xgb <- confusionMatrix(xgb_pred, test_data$continued_after_campaign)
cm_xgb

# 9. Konklusion til ML-model nr. 1 --------------------------------------------

# Vi har udviklet en klassifikationsmodel, der forudsiger, om en kunde 
# fortsætter efter kampagneperioden. Vi har brugt korrekt feature engineering og
# fjernet dataleakage, og opnået en XGBoost-model med en accuracy på 77,7%.
# Modellen identificerer 80% af churnerne og 75% af de kunder, der fortsætter, 
# hvilket gør den velegnet til at understøtte JPs problemstillinger. 


