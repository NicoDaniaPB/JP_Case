pacman::p_load(tidyverse, lubridate, caret, randomForest, xgboost, pROC)

# 1. Indlæsning af data -------------------------------------------------------

# Vi indlæser dataene ved "read_rds"-funktionen
model_data_raw <- read_rds("data/renset_datasæt.rds")

# 2. Yderligere klargøring af data til modellering ------------------------

# Filtrering til kunder med target 
model_data <- model_data_raw %>%
  filter(!is.na(continued_after_campaign))
# Vi beholder nu kun de kunder, hvor target-variablen "continued_after_campaign"
# ikke er NA. Altså kun observationer, hvor vi ved om kunden fortsatt efter 
# kampagnen. 

# Datoformatering
model_data <- model_data %>%
  mutate(
    order_date = as.Date(order_date),
    usr_created = as.Date(usr_created),
    subscription_cancel_date = as.Date(subscription_cancel_date),
    birthdate = as.Date(birthdate)
  )
# Vi har nu brugt mutate-funktionen til at bygge videre på "model_data", vi har
# omdannet fire kolonner til rigtige datoobjekter (Date). Dette trin er vigtigt
# senere for feature engineering og modellering.

# Vi fjerner order_tracktag og reason
model_data <- model_data %>%
  select(-order_trackertag, -reason)
# Vi har nu brugt mutate-funktionen til at bygge videre på "model_data", vi har
# fjernet de to kolonner, vi ikke kan bruge til modelleringsdelen. 

# Target som factor
model_data <- model_data %>%
  mutate(
    continued_after_campaign = factor(
      continued_after_campaign,
      levels = c(0, 1),
      labels = c("No", "Yes")
    )
  )
# Vi har nu brugt mutate-funktionen til at bygge videre på "model_data", vi har 
# lavet target-variablen om til en faktor med to niveauer, hvor: 
# 0 -> "No"
# 1 -> "Yes"

# 3. Feature engineering -----------------------------------------------------

# Vi skaber nu nye x-variabler, der kan forbedre modellens evne til at forudsige 
# y-variablen "continued_after_campaign"

model_data <- model_data %>%
  mutate(
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

# Vi har nu brugt mutate-funktionen til at bygge videre på "model_data", vi har
# bygget features, som kan forbedre modellens prediktionsevne. 

# 4. Leakage-variabler ----------------------------------------------------

# Vi fjerner nu leakage-variabler, dvs. variabler som ikke er kendt på det 
# tidspunkt, hvor vores model skal lave en forudsigelse. 

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
    -days_between_signup_and_order,
    -account_active_days,
    -churn_10
  )

# Vi har nu brugt pipe-operatoren til at bygge ovenpå "model_data". Vi har brugt
# select (og minus) fra tidyverse-pakken til at fjerne de bestemte 
# leakage-variabler.

# Vi fjerner pesudo_id lige før vi laver modellen
pseudo_ids <- model_data$pseudo_id
model_data <- model_data %>% select(-pseudo_id)

# 5. Train/test split --------------------------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi laver vores træning og test split (80/20 split i vores tilfælde)
train_index <- createDataPartition(model_data$continued_after_campaign, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

# Vi splitter pseudo_ids så det er med i test-dataene
test_ids <- pseudo_ids[-train_index]

# 6. Logistisk regression ----------------------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi definer cross-validation-kontrol 
cv_ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
# Vi har nu fortalt caret-pakken, at vi vil bruge en 5-fold cross-validation. Og
# at vi vil beregne klasse-sandsynligheder, og evaluere modellen med ROC-AUC. 
# Det sidstnævnte gøres via twoClassSummary. 

# Vi træner vores logistisk regression model (bruges til cross-validaiton) med 
# caret-pakken. 
log_cv <- train(
  continued_after_campaign ~ .,
  data = train_data,
  method = "glm",
  family = binomial,
  trControl = cv_ctrl,
  metric = "ROC"
)
# Vi har nu trænet alle features med "~", og brugt binomialt link, samt 
# fortalt caret-pakken, at vi vil bruge en 5-fold cross-validation. Og at vi vil
# evaluere modellen med ROC-AUC. 

# Nu træner vi en "ren" logistisk regression udet caret. Vi skal bruge den senere
# til at lave predictions på test-data. 
log_model <- glm(
  continued_after_campaign ~ ., 
  data = train_data,
  family = binomial
)
# Vi har nu trænet alle features med "~", og brugt binomialt link.

# Vi forudsiger nu sandsynglighederne på test-data
log_prob <- predict(log_model, test_data, type = "response")
# type = "response" giver sandsyngliheder for klassen "Yes"

# Vi laver nu vores ROC-kurve
roc_log <- roc(test_data$continued_after_campaign, log_prob)
# ROC-kurven er baseret på de sande labels, og de forudsagte sandsynligheder. 

# Vi finder det optimale cutoff (dvs. det cutoff, der maksimerer Youden's J)
cut_log <- as.numeric(coords(roc_log, "best", ret = "threshold"))
# Formlen er = sensitivitet + specificitet - 1. Dette giver bedste tradeoff
# mellem false positives og false negatives. 

# Vi konverterer nu sandsynligheder til klasser
log_pred <- ifelse(log_prob > cut_log, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))
# Sansynligheder over cutoff = "Yes", ellers er det "No". Det konverteres til 
# faktor med korrekt rækkefølge. 

# Vi laver vores Confusion matrix med test-dataene, for at evaluere modellens
# performance. 
cm_log <- confusionMatrix(log_pred, test_data$continued_after_campaign)
# Dette giver os vores møgletal, som skal bruges til at vurdere modellen. 

# 7. Random Forest -----------------------------------------------------------

# Vi sikrer reproducerbarhed 
set.seed(47)

# Vi laver en cross-validated Random Forest model med caret 
rf_cv <- train(
  continued_after_campaign ~ .,
  data = train_data,
  method = "rf",
  trControl = cv_ctrl,
  metric = "ROC",
  tuneLength = 5
)
# train() laver model tuning og cross-validation. 
# continued_after_campagin~. betyder at vi bruger alle variabler som predictors.
# method = "rf" gør at vi vælger Random Forest

# Vi træner nu en Random Forest model direkte uden cross-validation. Denne model
# skal bruges til prediction. 
rf_model <- randomForest(
  continued_after_campaign ~ .,
  data = train_data,
  ntree = 500,
  mtry = floor(sqrt(ncol(train_data) - 1)),
  importance = TRUE
)
# ntree = 500 -> sætter antallet af træer til 500
# mtry = sqrt(p) -> standardvalg for classification
# importrance = TRUE -> gemmer variabel vigtighed 

# Vi forudsiger nu sandsynligheder
rf_prob <- predict(rf_model, test_data, type = "prob")[,2]
# type = "prob" giver sandsynligheder
# [,2] henter sandsynligheden for klassen "Yes"

# Vi laver nu vores ROC-kurve
roc_rf <- roc(test_data$continued_after_campaign, rf_prob)
# ROC-kurven er baseret på de sande labels, og de forudsagte sandsynligheder. 

# Vi finder det optimale cutoff (dvs. det cutoff, der maksimerer Youden's J)
cut_rf <- as.numeric(coords(roc_rf, "best", ret = "threshold"))
# Formlen er = sensitivitet + specificitet - 1. Dette giver bedste tradeoff
# mellem false positives og false negatives. 

# Vi konverterer sandsynligheder til klasser
rf_pred <- ifelse(rf_prob > cut_rf, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))
# Hvis sandsynlighed > cutoff -> "Yes", ellers "No".
# Konverteres efterfølgende til en faktor med korrekt rækkefølge. 

# Vi laver vores Confusion matrix med test-dataene, for at evaluere modellens
# performance. 
cm_rf <- confusionMatrix(rf_pred, test_data$continued_after_campaign)
# Dette giver vores nøgletal.

# 8. XGBoost-----------------------------------------------------------------

# Vi laver vores model matrix
full_matrix <- model.matrix(continued_after_campaign ~ . - 1, data = model_data)
# model.matrix() konveerterer alle kategoriske variabler til dummy-variabler.
# -1 fjerner intercept.
# XGBoost kræver numeriske input -> derfor denne konvertering. 

# Vi laver vores test/train split i matrixen
train_matrix <- full_matrix[train_index, ]
test_matrix  <- full_matrix[-train_index, ]
# train_index bruges til at vælge rækker til træning.
# Resten bruges som testdata. 

# Vi laver labels til træning
y_train <- model_data$continued_after_campaign[train_index]
# Vi henter target-variablen for træningssættet. 

# Vi opretter DMatrix (XGBoosts eget format)
dtrain <- xgb.DMatrix(data = train_matrix, label = as.numeric(y_train) - 1)
dtest  <- xgb.DMatrix(data = test_matrix,  label = as.numeric(model_data$continued_after_campaign[-train_index]) - 1)
# XGBoost kræver labels som 0/1, ikke kategorier som "Yes/No". Derfor sætter vi
# as.numeric(...) - 1. 

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi laver cross-validation for XGBoost
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
# nfold = 5 -> kører 5-fold cross-validation. 
# nrounds = 3000 -> evaluerer modellen over 300 boosting-runder.
# eval_metrix = "auc" -> måler AUC for hver fold.
# Vi bruger cross-validation til at se, om modellen overfitter.

# Vi træner den endelige XGBoost-model 
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
# Vi træner en XGBoost-model med de samme hyperparamtere som tidligere.
# Ingen tuning her, bare en fast model. 

# Vi forudsiger sandsynligheder
xgb_prob <- predict(xgb_model, dtest)
# Outputtet bliver en sandsynglighed for klassen 1 ("Yes").

# Vi laver nu vores ROC-kurve
roc_xgb <- roc(test_data$continued_after_campaign, xgb_prob)
# ROC-kurven er baseret på de sande labels, og de forudsagte sandsynligheder. 

# Vi finder det optimale cutoff (dvs. det cutoff, der maksimerer Youden's J)
cut_xgb <- as.numeric(coords(roc_xgb, "best", ret = "threshold"))
# Formlen er = sensitivitet + specificitet - 1. Dette giver bedste tradeoff
# mellem false positives og false negatives. 

# Vi konverterer sandsynligheder til klasser
xgb_pred <- ifelse(xgb_prob > cut_xgb, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))
# Hvis sandsynlighed > cutoff -> "Yes", ellers "No".
# Konverteres efterfølgende til en faktor med korrekt rækkefølge. 

# Vi laver vores Confusion matrix med test-dataene, for at evaluere modellens
# performance. 
cm_xgb <- confusionMatrix(xgb_pred, test_data$continued_after_campaign)
# Dette giver vores nøgletal

# 9. Tabel over mest risikable kunder -------------------------------------

# BEMÆRK: der 211, fordi det er test_dataene vi har gemt pseudo_ids i. 
# Vi opretter en tabel med ID, sandsynlighed og forudsagt klasse
risk_list <- tibble(
  pseudo_id = test_ids,
  churn_probability = xgb_prob,
  predicted_class = xgb_pred
)
# pseudo_id = de ID'er vi gemte tidligere fra test_data (derfor 211 rækker).
# churn_probability = sandsynligheden for churn fra XGBoost. 
# predicted_class = "Yes"/"No" baseret på ROC-cutoff.
risk_list_sorted <- risk_list %>%
  arrange(desc(churn_probability))
# Resultatet bliver en tabel med en række pr. testkunde. 

# Vi ser resultatet
head(risk_list_sorted, 50)

# Vi sorter kunder efter højeste churn-risiko
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
# Vi har nu sorteret kunderne fra højeste til laveste risiko for churn. 

# Vi ser resultaterne
head(risk_table, 50)
view(risk_table)

# 10. Økonomisk analyse af churn-kampagne -------------------------------------

# Vi laver nu en økonomisk beregning af værdien ved at redde høj-risiko-kunder
# og vi gør det for tre forskellige scenarier. 

# Listepris efter kampagne 
JP_pris <- 199  

# Definition af højrisiko-kunder
high_risk <- risk_list_sorted %>% 
  filter(churn_probability > 0.75)
n_high_risk <- nrow(high_risk)
# Vi filtrerer kunder med en churn-sandsynlighed over 75%.
# n_high_risk = antal højrisiko-kunder. 

# 5% uplift
uplift_5 <- 0.05
saved_5 <- n_high_risk * uplift_5
value_3m_5  <- saved_5 * JP_pris * 3
value_6m_5  <- saved_5 * JP_pris * 6
value_12m_5 <- saved_5 * JP_pris * 12

# 15% uplift
uplift_15 <- 0.15
saved_15 <- n_high_risk * uplift_15
value_3m_15  <- saved_15 * JP_pris * 3
value_6m_15  <- saved_15 * JP_pris * 6
value_12m_15 <- saved_15 * JP_pris * 12

# 25% uplift
uplift_25 <- 0.25
saved_25 <- n_high_risk * uplift_25
value_3m_25  <- saved_25 * JP_pris * 3
value_6m_25  <- saved_25 * JP_pris * 6
value_12m_25 <- saved_25 * JP_pris * 12

# Samlet tabel
economy_table <- tibble(
  scenario = c("5% (Worst case)", "15% (Realistisk)", "25% (Optimistisk)"),
  højrisiko_kunder = n_high_risk,
  reddede_kunder = round(c(saved_5, saved_15, saved_25)),
  besparelse_3m = round(c(value_3m_5, value_3m_15, value_3m_25)),
  besparelse_6m = round(c(value_6m_5, value_6m_15, value_6m_25)),
  besparelse_12m = round(c(value_12m_5, value_12m_15, value_12m_25))
)
# Vi har nu samlet resultaterne i en tibble ved combine (c) funktionen. 

# Vi ser resultaterne
economy_table

# 11. AUC-sammenligning -------------------------------------------------------

# Vi beregner AUC for alle modeller 
auc_log <- as.numeric(pROC::auc(roc_log))
auc_rf  <- as.numeric(pROC::auc(roc_rf))
auc_xgb <- as.numeric(pROC::auc(roc_xgb))
# auc() udtrækker AUC-værdien.
# as.numeric() siker at værdien er et tal. 

# Vi laver nu en sammenlignignstabel
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
# Vi har lavet en tibble med en række pr. model
# Vi henter accuracy fra cm_*overall, sensitivity og specificty fra cm*$byClass.
# Alle værdier afrundes til 4 decimaler. 

# Vi printer resultaterne i konsolen
print(model_sammenligning)

# Vi laver et plot over ROC-kurverne
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
# Første plot() laver figuren. 
# De næste to bruger add = TRUE -> tilføjer kurverne ovenpå.
# Vi farvelægger modellerne for at gøre det lettere at skelne. 
# Vi tilføjer legend, hvilket viser modelnavne + AUC direkte i figuren. 
# Farverne matcher kurverne. lwd = 2 giver tykkere linjer. 

# 12. Variabelvigtighed -------------------------------------------------------

# Random Forest: built-in importance
# MeanDecreaseGini måler hvor meget hver variabel bidrager til
# at reducere usikkerhed på tværs af alle træer i skoven
rf_imp <- importance(rf_model) %>%
  as.data.frame() %>%
  rownames_to_column("variabel") %>%
  arrange(desc(MeanDecreaseGini))
# importance(rf_model) udtrækker RF's indbyggede variable importance matrix.
# MeanDecreaseGini er RF's standardmål. Måler hvor meget en variabel reducerer
# Gini-impurity på tværs af alle træer. Jo højere værdi, jo vigtigere er 
# variablen. 
# rownames_to_column("varaibel") gør variabelnavne til en kolonne.
# arrange(desc(...)) sortrer efter de vigtigste variabler. 

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
# Dette viser de 15 vigtigste variabler for RF.

# XGBoost: feature importance
xgb_imp <- xgb.importance(model = xgb_model) %>%
  as_tibble()
# xgb.importance() udtrlkker XGBoost's importance scores.
# Standardmålet her er "Gain". Dvs hvor meget modellen forbedrer modellens 
# loss-funktion. 

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
# Dette viser de 15 vigtigste variabler for XGBoost.

# 13. Cross validation sammenligning --------------------------------------

# Udtræk af CV-resultater fra logistisk regression via caret-pakken
log_cv_results <- log_cv$results %>%
  select(ROC, Sens, Spec) %>%
  slice(which.max(ROC)) %>%
  mutate(Model = "Logistisk regression")
# log_cv$results indeholder alle CV‑kombinationer fra caret.
# select(ROC, Sens, Spec) henter de metrics vi vil sammenligne.
# slice(which.max(ROC)) vælger den række (tuning‑kombination) med højeste ROC.
# mutate(Model = "...") tilføjer modelnavn.

# Udtræk af CV-resultater fra Random Forest via caret-pakken
rf_cv_results <- rf_cv$results %>%
  select(ROC, Sens, Spec) %>%
  slice(which.max(ROC)) %>%
  mutate(Model = "Random Forest")
# Samme som ved logistisk regression.

# Udtræk  af CV-resultater fra XGBoost via caret-pakken
xgb_cv_results <- tibble(
  Model = "XGBoost",
  ROC   = max(xgb_cv2$evaluation_log$test_auc_mean),
  Sens  = NA,   # xgb.cv giver ikke Sens/Spec direkte
  Spec  = NA
)
# XGBoosts egen CV xgb.cv()giver kun AUC, ikke Sens/Spec.
# Derfor sætter vi Sens og Spec til NA.
# max(xgb_cv2$evaluation_log$test_auc_mean) finder den bedste AUC fra 
# CV‑forløbet.

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
# 87.28%. Hvilket betyder, at det er den der generaliser bedst på nye ukendte 
# data. Random Forest performer også meget godt, men en CV-ROC på 87.12%. 

# 14. Konklusion til ML-model nr. 1 --------------------------------------------

# Vi har udviklet en klassifikationsmodel, der forudsiger, om en kunde 
# fortsætter efter kampagneperioden. Vi har brugt korrekt feature engineering og
# fjernet dataleakage, og lavet tre forskellige klassifikationsmodeller. XGBoost
# modellen opnår den højeste AUC på 90.1%. og dermed den bedste balance mellem 
# sensitivity og specificity, hvor den finder 90.1% true-positives. Det er 
# derfor den bedste model til at forudsige churn efter kampagnen. 
# XGBoost-model har en accuracy på 82.5%. Modellen identificerer 90.4% af 
# churnerne og 74.8% af de kunder, der fortsætter, hvilket gør den velegnet til 
# at understøtte JPs problemstillinger. 

# 15. Gem som CSV-filer til Power BI ------------------------------------------

saveRDS(xgb_model, "data/xgb_model.rds")

# Risiko liste
risk_list %>%
  mutate(across(where(is.numeric), ~ round(.x, 4))) %>%
  write.table("data/risiko_liste.csv",
              sep = ";", dec = ",", row.names = FALSE, quote = FALSE)

# Økonomisk besparelse
write.csv(economy_table, "data/økonomisk_besparelse.csv", row.names = FALSE)

# Variabelvigtighed
xgb_imp %>%
  mutate(across(where(is.numeric), ~ round(.x, 4))) %>%
  write.table("data/xgb_variabel_vigtighed.csv",
              sep = ";", dec = ",", row.names = FALSE, quote = FALSE)
