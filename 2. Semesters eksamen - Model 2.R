pacman::p_load(tidyverse, lubridate, caret, pROC, randomForest, xgboost)

# Vi har nu loaded de pakker vi skal bruge. Vi bruger tidyverse til 
# datamanipulation. lubridate bruges til at håndterer datoer. caret bruges til 
# train/test split og modelværktøjer. pROC bruges til ROC-kurver og cutoffs i 
# vores klassifikationsmodeller. randomForest- og xgboost- pakkerne bruges til
# at køre RF og Boosting modellerne.

# Vores anden model skal kunne klassificere, om kunder der forsætter efter
# kampagneperioden churner hurtigt eller ej

# 1. Indlæsning af data ---------------------------------------
model_data <- read_rds("data/renset_datasæt.rds")

# 2. Yderligere klargøring af data til modellerne -------------------------

# Filtrering til kunder der FORTSÆTTER efter kampagnen
model2_data <- model_data %>%
  filter(continued_after_campaign == 1)

# Target-variabel: fast churn 
model2_data <- model2_data %>%
  mutate(
    fast_churn = as.factor(fast_churn)
  )

# 3. Feature engineering --------------------------------------------------

# Vi opretter nu nogle nye features, som skal være med til at øge modellens 
# forklaringskraft. Udover det, så fjerner vi også NA-værdierne. 
model2_data <- model2_data %>%
  mutate(
    days_active_before_campaign = account_active_days,
    days_since_user_created = as.numeric(order_date - usr_created),
    days_between_signup_and_order = as.numeric(order_date - usr_created),
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


# 4. Fjerning af lækage-variabler ------------------------------------------

model2_data <- model2_data %>%
  select(
    -subscription_cancel_date,
    -end_date,
    -expiration_date,
    -subscription_length_days,
    -length_group,
    -pseudo_id,
    -order_trackertag,
    -reason
  )

# Vi har nu fjerner de variabler, som indeholder infromation om fremtiden
# (fx opsigelsesdato). Dette forhinder dataleakage, og sikrer at vi har en 
# realistisk model, som kan anvendes i praksis. Vi bruger select til at fjerne
# variablerne. 

# 5. Train/test split -----------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi laver et 80/20 split 
train_index2 <- createDataPartition(model2_data$fast_churn, p = 0.8, list = FALSE)
train_data2 <- model2_data[train_index2, ]
test_data2  <- model2_data[-train_index2, ]

# 6. Logistisk regression -------------------------------------

# Vi træner en logistisk regressionsmodel på alle vores variabler
log_model2 <- glm(
  fast_churn ~ .,
  data = train_data2,
  family = binomial
)

# Prediction - Vi forudsiger sandsynligheder for vores testdatasæt
log_prob2 <- predict(log_model2, test_data2, type = "response")

# Vi bereger ROC-kurven og finder det optimale cutoff
roc_log2 <- roc(test_data2$fast_churn, log_prob2)
cut_log2 <- as.numeric(coords(roc_log2, "best", ret = "threshold"))

# Vi konverterer sandsynlighederne til klasser og beregner confusion matrix
log_pred2 <- ifelse(log_prob2 > cut_log2, "1", "0") %>% factor(levels = c("0","1"))
cm_log2 <- confusionMatrix(log_pred2, test_data2$fast_churn)
cm_log2

# 7. Random Forest --------------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi laver vores Random Forest model
rf_model2 <- randomForest(
  fast_churn ~ .,
  data = train_data2,
  ntree = 500,
  mtry = floor(sqrt(ncol(train_data2) - 1)),
  importance = TRUE
)

# Prediction - Vi forudsiger sandsynligheder for vores testdatasæt
rf_prob2 <- predict(rf_model2, test_data2, type = "prob")[,2]

# Vi bereger ROC-kurven og finder det optimale cutoff
roc_rf2 <- roc(test_data2$fast_churn, rf_prob2)
cut_rf2 <- as.numeric(coords(roc_rf2, "best", ret = "threshold"))

# Klassifikation
rf_pred2 <- ifelse(rf_prob2 > cut_rf2, "1", "0") %>% factor(levels = c("0","1"))

# Vi laver vores Confusion Matrix
cm_rf2 <- confusionMatrix(rf_pred2, test_data2$fast_churn)
cm_rf2

# 8. XGBoost ---------------------------------------------------

# Vi laver en samlet model.matrix for hele datasættet
full_matrix2 <- model.matrix(fast_churn ~ . - 1, data = model2_data)

# Vi konverterer target til numerisk
full_label2  <- as.numeric(model2_data$fast_churn) - 1

# Vi splitter matrix og labels i train/test
train_matrix2 <- full_matrix2[train_index2, ]
test_matrix2  <- full_matrix2[-train_index2, ]

train_label2 <- full_label2[train_index2]
test_label2  <- full_label2[-train_index2]

# Vi laver en DMatrix
dtrain2 <- xgb.DMatrix(data = train_matrix2, label = train_label2)
dtest2  <- xgb.DMatrix(data = test_matrix2,  label = test_label2)

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi træner vores XGBoost-model
xgb_model2 <- xgb.train(
  data = dtrain2,
  nrounds = 300,
  max_depth = 4,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  objective = "binary:logistic",
  eval_metric = "auc"
)

# Prediction 
xgb_prob2 <- predict(xgb_model2, dtest2)

# ROC og cutoff
roc_xgb2 <- roc(test_data2$fast_churn, xgb_prob2)
cut_xgb2 <- as.numeric(coords(roc_xgb2, "best", ret = "threshold"))

# Klassifikation
xgb_pred2 <- ifelse(xgb_prob2 > cut_xgb2, "1", "0") %>% factor(levels = c("0","1"))

# Vi laver vores Confusion Matrix
cm_xgb2 <- confusionMatrix(xgb_pred2, test_data2$fast_churn)
cm_xgb2

# 9. Konklusion til model nr. 2 -------------------------------------------

# Vi har udviklet en klassifikationsmodel, der forudsiger, om hvilke kunder der
# fortsætter efter kampagneperioden churner hurtigt eller ej.
# Vi har brugt korrekt feature engineering og fjernet dataleakage, og opnået en
# XGBoost-model med en accuracy på 85%.
# Modellen identificerer 91% af dem der churner hurtigt (sensitivity) og 69% af 
# de kunder der faktsik churner hurtgit. Modellen fanger derfor 7/10 risikokunder.
# Midellen vurderes derfor velegnet til at understøtte JPs problemstillinger. 
  
