pacman::p_load(tidyverse, lubridate, caret, pROC, randomForest, xgboost)

# Vi har nu loaded de pakker vi skal bruge. Vi bruger tidyverse til 
# datamanipulation. lubridate bruges til at håndterer datoer. caret bruges til 
# train/test split og modelværktøjer. pROC bruges til ROC-kurver og cutoffs i 
# vores klassifikationsmodeller. randomForest- og xgboost- pakkerne bruges til
# at køre RF og Boosting modellerne.

# Vores anden model skal kunne klassificere, om kunder der fortsætter efter
# kampagneperioden churner inden for 10 dage eller ej.
# Argumentationen for 10 dage: abonnementet er forudbetalt, og kunder der
# glemte at afmelde vil typisk opdage det når de ser første betaling og
# afmelde inden for de første 10 dage efter kampagnens afslutning.

# 1. Indlæsning af data ---------------------------------------

# Vi indlæser dataene ved "read_rds"-funktionen
model_data <- read_rds("data/renset_datasæt.rds")

# Vi gemmer ID'er separat
all_ids <- model_data$pseudo_id
# Vi gør dette, så vi kan lave risiko-liste senere i koden. 

# 2. Yderligere klargøring af data til modellerne -------------------------

# Filtrering til kunder der FORTSÆTTER efter kampagnen
model2_data <- model_data %>%
  filter(continued_after_campaign == 1)
# Vi har filtereret dataene, så vi kun har de kunder med der forsætter efter 
# kampagnen. Vi bruger pipeline operatoren til at bygge videre på "model_data"
# til at filtrere hvor "continued_after_campaign" er = 1. Dvs kun de kunder, 
# der forsætter efter kampagnen. 

# Vi gemmer ID'er for Model 2 datasættet
model2_ids <- model2_data$pseudo_id
# Vi tager "pseudo_id" fra "model2_data" og gemmer det i en separat vektor vi 
# kalder "model2_ids".

# Target-variabel: churn_10
# Vi konverterer til factor med gyldige labels til caret
model2_data <- model2_data %>%
  mutate(
    churn_10 = factor(
      churn_10,
      levels = c(0, 1),
      labels = c("No", "Yes")
    )
  )
# Vi bruger pipeline-operatoren til at bygge videre på "model2_data". Vi bruger
# mutate til at konverterer "churn_10" til en factor med niveauerne 0 og 1. 
# Dvs. churner eller ikke-churner. 

# 3. Feature engineering --------------------------------------------------

# Vi opretter nu nogle nye features, som skal være med til at øge modellens 
# forklaringskraft.
# Vi bruger pipeline-operatoren til at bygge videre på "model2_data". Vi bruger
# mutate til at bygge nogle features, som skal styrke modellens forklaringsgrad.
model2_data <- model2_data %>%
  mutate(
    days_since_user_created = as.numeric(difftime(order_date, usr_created, units = "days")),
    age_at_order = as.numeric(difftime(order_date, birthdate, units = "days")) / 365
  )

# 4. Udvælgelse af variabler ------------------------------------------
model2_data <- model2_data %>%
  select(
    # Demografiske
    koen,
    age_at_order,
    
    # Abonnementshistorik
    previous_subscriptions,
    previous_campaigns,
    previous_trials,
    account_active_days_before_campaign,
    
    # Permission og nyhedsbreve
    permission_given_order,
    newsletters_before_order,
    newsletters_after_order,
    
    # Adfærd de første 30 dage (ingen leakage – kampagnen er 2 måneder)
    visits,
    unique_pages,
    restricted_views,
    restricted_ratio,
    avg_scroll,
    mobile_ratio,
    desktop_ratio,
    
    # Feature engineered
    days_since_user_created,
    
    # Target
    churn_10
  ) 

# Vi har kun beholdt variabler som er kendte ved eller kort efter køb.
# permission_given_today er fjernet da den er målt pr. 10. marts 2026 (fremtidig info).
# Adfærdsvariablerne er målt de første 30 dage efter køb, hvilket er inden
# kampagnens udløb, og giver derfor ikke dataleakage.

# 5. Train/test split -----------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Vi laver et 80/20 split 
train_index2 <- createDataPartition(model2_data$churn_10, p = 0.8, list = FALSE)
train_data2 <- model2_data[train_index2, ]
test_data2  <- model2_data[-train_index2, ]

# 6. Håndtering korrelerede variabler -----------------------------------------

# Vi vil nu håndtere lineære korrelerede variabler, da lineære modeller som 
# logistisk regression ikke kan håndtere det selv. Træ-baserede modeller som 
# Random Forest eller XGBoost kan godt håndtere det, så derfor gør vi det kun
# for vores logistisk regression model.

# Tjek for perfekt korrelerede variabler
cor_matrix2 <- cor(train_data2 %>% select(where(is.numeric)))
high_cor2 <- which(abs(cor_matrix2) > 0.95 & cor_matrix2 != 1, arr.ind = TRUE)
print(high_cor2)

# Tjek for lineær afhængighed
findLinearCombos(train_data2 %>% select(where(is.numeric)))

# Vi har nu kontrolleret for lineære afhænigheder og behøver ikke foretage os 
# yderligere tiltag, da der ingen er. 

# 7. Logistisk regression -------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Cross validation
cv_ctrl2 <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Vi definerer hvordan caret skal validere modellen:
# method = "cv" → almindelig k‑fold cross‑validation
# number = 5 → 5 fold
# classProbs = TRUE → caret skal beregne sandsynligheder (ikke kun klasser)
# summaryFunction = twoClassSummary → caret skal optimere efter:
# ROC, sensitivitet og specificitet
# Dette kræver at target‑variablen er en factor med levels: "No" og "Yes".

log_cv2 <- train(
  churn_10 ~ .,
  data = train_data2,
  method = "glm",
  family = binomial,
  trControl = cv_ctrl2,
  metric = "ROC"
)
# caret træner en logistisk regression (glm)
# alle variabler bruges som features (~ .)
# cross‑validation styres af cv_ctrl2
# modellen optimeres efter ROC
# log_cv2 indeholder:
# CV‑ROC, CV‑sensitivitet og CV‑specificitet, koefficienter og performance på
# tværs af fold.

# Vi ser resultatet
log_cv2

# Vi træner en logistisk regressionsmodel på alle vores variabler
log_model2 <- glm(churn_10 ~ ., data = train_data2, family = binomial)
# Vi træner her en klassisk logistisk regression uden cross‑validation.
# Grunden til vi gør begge dele er følgende: 
# caret‑modellen bruges til performance‑estimat og glm‑modellen bruges til 
# prediction og ROC‑analyse på testdata.

# Prediction - Vi forudsiger sandsynligheder for vores testdatasæt
log_prob2 <- predict(log_model2, test_data2, type = "response")
# modellen forudsiger sandsynligheder for churn, og output er tal mellem 0 og 1.

# Vi bereger ROC-kurven og finder det optimale cutoff
roc_log2 <- roc(test_data2$churn_10, log_prob2)
cut_log2 <- as.numeric(coords(roc_log2, "best", ret = "threshold"))
# roc() beregner ROC‑kurven
# coords(..., "best") finder det cutoff der maksimerer Youden's J, hvilket er:
# = sensitivitet + specificitet – 1. Det giver os det optimale threshold i 
# stedet for standard 0.5.

# Vi konverterer sandsynlighederne til klasser og beregner confusion matrix
log_pred2 <- ifelse(log_prob2 > cut_log2, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))
cm_log2 <- confusionMatrix(log_pred2, test_data2$churn_10)
cm_log2

# Sandsynligheder → klasser:
# cutoff → "Yes" (churn)
# ellers → "No"
# Vi konverterer til factor med korrekt rækkefølge, caret kræver at "No" er
# første level. confusionMatrix() beregner vores nøgletal.

# 8. Random Forest --------------------------------------------

# Vi sikrer reproducerbarhed
set.seed(47)

# Cross validation
rf_cv2 <- train(
  churn_10 ~ .,
  data = train_data2,
  method = "rf",
  trControl = cv_ctrl2,
  metric = "ROC",
  tuneLength = 5
)
# caret træner en Random Forest‑model
# ~ . betyder: brug alle features
# method = "rf" → caret bruger randomForest‑pakken
# trControl = cv_ctrl2 → 5‑fold cross‑validation
# metric = "ROC" → caret optimerer efter ROC‑score
# tuneLength = 5 → caret tester 5 forskellige værdier af mtry 
# (antal variabler pr. split)

# Vi ser resultatet: 
rf_cv2

# Vi laver vores Random Forest model
rf_model2 <- randomForest(
  churn_10 ~ .,
  data = train_data2,
  ntree = 500,
  mtry = floor(sqrt(ncol(train_data2) - 1)),
  importance = TRUE
)
# ntree = 500 -> antal træer i skoven
# mtry = sqrt(p) -> klassisk tommelfingerregel for klassifikation
# importance = TRUE → modellen beregner variable importance
# caret‑modellen bruges til cross‑validation performance
# den manuelle model bruges til prediction og ROC‑analyse på testdata

# Prediction - Vi forudsiger sandsynligheder for vores testdatasæt
rf_prob2 <- predict(rf_model2, test_data2, type = "prob")[,2]
# Random Forest forudsiger klasse‑sandsynligheder
# [,2] henter sandsynligheden for klassen "Yes" (churn)

# Vi bereger ROC-kurven og finder det optimale cutoff
roc_rf2 <- roc(test_data2$churn_10, rf_prob2)
cut_rf2 <- as.numeric(coords(roc_rf2, "best", ret = "threshold"))
# ROC‑kurven beregnes, og det optimale cutoff findes ud fra Youden's J:
# max(sensitivitet + specificitet – 1)
# Dette giver et bedre cutoff end standard 0.5.

# Klassifikation
rf_pred2 <- ifelse(rf_prob2 > cut_rf2, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))
# Sandsynligheder konverteres til klasser:
# cutoff -> "Yes"
# ellers -> "No"
# Factor levels sættes korrekt (caret kræver "No" først)

# Vi laver vores Confusion Matrix
cm_rf2 <- confusionMatrix(rf_pred2, test_data2$churn_10)
cm_rf2
# Vi har nu en fuld evaluering af modellen. 

# 9. XGBoost ---------------------------------------------------

# Vi laver en samlet model.matrix for hele datasættet
full_matrix_m2 <- model.matrix(churn_10 ~ . - 1, data = model2_data)
colnames(full_matrix_m2)  
# model.matrix() konverterer alle features til numeriske kolonner, inkl. 
# one‑hot encoding af faktorer.
# -1 fjerner intercept‑kolonnen.
# XGBoost kræver rene numeriske matricer, så dette er nødvendigt.

# Vi konverterer target til numerisk
full_label_m2 <- as.numeric(model2_data$churn_10) - 1
# churn_10 er en factor med levels "No" og "Yes".
# as.numeric() gør dem til 1 og 2.
# -1 gør dem til 0 og 1 (som XGBoost kræver).

# Vi splitter matrix og labels i train/test
train_matrix_m2 <- full_matrix_m2[train_index2, ]
test_matrix_m2  <- full_matrix_m2[-train_index2, ]
# Vi splitter både features og labels i train/test.
# Vi gemmer både:
# numeriske labels (til XGBoost)
# factor labels (til caret‑lignende funktioner).

# Brug den originale factor-target til CV
y_train_m2 <- model2_data$churn_10[train_index2]

train_label_m2 <- full_label_m2[train_index2]
test_label_m2  <- full_label_m2[-train_index2]

# Vi laver en DMatrix
dtrain_m2 <- xgb.DMatrix(data = train_matrix_m2, label = as.numeric(y_train_m2) - 1)
dtest_m2  <- xgb.DMatrix(data = test_matrix_m2,
                         label = as.numeric(model2_data$churn_10[-train_index2]) - 1)
# XGBoost bruger sin egen datastruktur: DMatrix.
# Den er optimeret til hurtig træning og mindre hukommelsesforbrug.
# Labels konverteres igen til 0/1.

# Vi sikrer reproducerbarhed
set.seed(47)

# Cross validation
xgb_cv_m2 <- xgb.cv(
  params = list(
    max_depth = 4,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = "binary:logistic",
    eval_metric = "auc"
  ),
  data = dtrain_m2,
  nrounds = 300,
  nfold = 5,
  verbose = 0
)
# Vi kører 5‑fold cross‑validation med:
# max_depth = 4 → hvor dybe træerne må være
# eta = 0.05 → learning rate
# subsample = 0.8 → hvor stor en del af data bruges pr. træ
# colsample_bytree = 0.8 → hvor mange features bruges pr. træ
# objective = "binary:logistic" → binær klassifikation
# eval_metric = "auc" → optimer efter AUC

# Vi ser resultatet:
xgb_cv_m2

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
  data = dtrain_m2,
  nrounds = 300
)
# Vi træner en fuld XGBoost‑model på hele træningsdatasættet.
# Parametrene er de samme som i CV.
# nrounds = 300 betyder 300 boosting‑iterationer.

# Prediction 
xgb_prob2 <- predict(xgb_model2, dtest_m2)
# XGBoost returnerer sandsynligheder for klassen "Yes".

# ROC og cutoff
roc_xgb2 <- roc(test_data2$churn_10, xgb_prob2)
cut_xgb2 <- as.numeric(coords(roc_xgb2, "best", ret = "threshold"))
# ROC‑kurven beregnes, og det optimale cutoff findes via Youden's J:
# max = (sensitivitet + specificitet – 1)
# Dette giver et bedre cutoff end 0.5.

# Klassifikation
xgb_pred2 <- ifelse(xgb_prob2 > cut_xgb2, "Yes", "No") %>%
  factor(levels = c("No", "Yes"))
# Sandsynligheder → klasser
# Factor levels sættes korrekt (caret kræver "No" først)

# Vi laver vores Confusion Matrix
cm_xgb2 <- confusionMatrix(xgb_pred2, test_data2$churn_10)
cm_xgb2
# Vi har nu en fuld evaluering af modellen. 

# 10. Tabel over risikable kunder ------------------------------------------

# Gem ID'er for datasættene
train_ids_m2 <- model2_ids[train_index2]
test_ids_m2  <- model2_ids[-train_index2]
# Vi tager de pseudo‑ID'er, vi gemte tidligere (model2_ids)
# Vi splitter dem i:
# train_ids_m2 → ID'er i træningsdatasættet
# test_ids_m2 → ID'er i testdatasættet
# Det er fordi vi laver predictions på testdata, og vi skal kunne matche dem 
# tilbage til de rigtige kunder.

# Lav risk-listen med de rigtige variabler
risk_list_m2 <- tibble(
  pseudo_id = test_ids_m2,
  churn_probability = xgb_prob2,
  predicted_class = xgb_pred2
)
# Vi bygger en tabel med:
# pseudo_id → kundens ID.
# churn_probability → sandsynlighed for churn (fra XGBoost).
# predicted_class → "Yes" eller "No" baseret på cutoff.
# Dette er den rå risikoliste.

# Sortér efter risiko
risk_list_sorted_m2 <- risk_list_m2 %>%
  arrange(desc(churn_probability))
# Listen sorteres fra højeste til laveste churn‑sandsynlighed.
# De mest risikofyldte kunder kommer øverst.

# Lav endelig tabel
risk_table_m2 <- risk_list_sorted_m2 %>%
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
# Vi laver den endelige tabel med tre risikogrupper:
# Højere end 75% churn-risiko
# Mellem 50% til 75% churn-risiko
# Under 50% churn risiko

# 11. Økonomisk analyse af churn-kampagne ---------------------------------

# Vi laver nu en økonomisk beregning af værdien ved at redde høj-risiko-kunder
# og vi gør det for tre forskellige scenarier. 

# Listepris efter kampagne 
JP_pris <- 199  

# Definition af højrisiko-kunder
high_risk_m2 <- risk_list_sorted_m2 %>% 
  filter(churn_probability > 0.75)
n_high_risk_m2 <- nrow(high_risk_m2)
# Vi filtrerer kunder med en churn-sandsynlighed over 75%.
# n_high_risk_m2 = antal højrisiko-kunder. 

# 5% uplift
saved_5_m2   <- n_high_risk_m2 * 0.05
value_3m_5_m2  <- saved_5_m2 * JP_pris * 3
value_6m_5_m2  <- saved_5_m2 * JP_pris * 6
value_12m_5_m2 <- saved_5_m2 * JP_pris * 12

# 15% uplift
saved_15_m2   <- n_high_risk_m2 * 0.15
value_3m_15_m2  <- saved_15_m2 * JP_pris * 3
value_6m_15_m2  <- saved_15_m2 * JP_pris * 6
value_12m_15_m2 <- saved_15_m2 * JP_pris * 12

# 25% uplift
saved_25_m2   <- n_high_risk_m2 * 0.25
value_3m_25_m2  <- saved_25_m2 * JP_pris * 3
value_6m_25_m2  <- saved_25_m2 * JP_pris * 6
value_12m_25_m2 <- saved_25_m2 * JP_pris * 12

# Samlet tabel
economy_table_m2 <- tibble(
  scenario = c("5% (Worst case)", "15% (Realistisk)", "25% (Optimistisk)"),
  højrisiko_kunder = n_high_risk_m2,
  reddede_kunder = round(c(saved_5_m2, saved_15_m2, saved_25_m2)),
  besparelse_3m = round(c(value_3m_5_m2, value_3m_15_m2, value_3m_25_m2)),
  besparelse_6m = round(c(value_6m_5_m2, value_6m_15_m2, value_6m_25_m2)),
  besparelse_12m = round(c(value_12m_5_m2, value_12m_15_m2, value_12m_25_m2))
)
# Vi har nu samlet resultaterne i en tibble ved combine (c) funktionen. 

# Vi ser resultaterne
economy_table_m2

# 12. AUC-sammenligning -------------------------------------------------------

# Vi beregner AUC for alle modeller
auc_log2 <- as.numeric(pROC::auc(roc_log2))
auc_rf2  <- as.numeric(pROC::auc(roc_rf2))
auc_xgb2 <- as.numeric(pROC::auc(roc_xgb2))
# Vi tager ROC‑objekterne for hver model.
# Vi beregner deres AUC (Area Under the Curve).
# Vi konverterer dem til numeriske værdier.
# AUC er et mål for modellens evne til at skelne mellem churn og ikke‑churn.

# Vi sammenligner modellerne ved at samle dem i en tibble
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
# Alle værdier hentes direkte fra confusion matrices og ROC‑objekter.
# Dette giver os en fuld performance‑sammenligning.

# Vi udskriver tabellen
print(model_sammenligning2)

# Vi plotter ROC-kurverne
plot(roc_log2, col = "steelblue", lwd = 2,
     main = "ROC-kurver — Model 2 (churn_10 efter kampagneslut)")
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
# Først plottes ROC‑kurven for logistisk regression, derefter tilføjes 
# Random Forest og XGBoost ovenpå. Farverne gør det nemt at skelne modellerne.
# Dette giver et visuelt overblik over hvilken model der performer bedst.
# Til sidst tilføjer vi en forklaringsboks nederst til højre, den viser: 
# modelnavn, farve og AUC‑værdien. Dette gør grafen let at aflæse. 

# 13. Variabelvigtighed -------------------------------------------------------

# Random Forest: built-in importance
# MeanDecreaseGini måler hvor meget
# at reducere usikkerhed på tværs af alle træer i skoven
rf_imp2 <- importance(rf_model2) %>%
  as.data.frame() %>%
  rownames_to_column("variabel") %>%
  arrange(desc(MeanDecreaseGini))
# importance(rf_model2) henter Random Forest's indbyggede variabelvigtighed.
# Random Forest bruger MeanDecreaseGini, som måler hvor meget en variabel 
# reducerer usikkerhed (Gini impurity) på tværs af alle træer.
# Vi konverterer resultatet til en data frame.
# Vi tilføjer variabelnavne som en kolonne.
# Vi sorterer variablerne efter vigtigst → mindst vigtig.

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
# Vi tager de 15 vigtigste variabler.
# Vi laver et horisontalt søjlediagram.
# Variablerne sorteres efter betydning.

# XGBoost: feature importance
xgb_imp2 <- xgb.importance(model = xgb_model2) %>%
  as_tibble()
# xgb.importance() henter XGBoost's feature importance.
# XGBoost bruger typisk Gain, som måler hvor meget en variabel forbedrer 
# modellens præcision, når den bruges til splits.

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
# Vi tager de 15 vigtigste XGBoost‑features.
# Vi laver et horisontalt søjlediagram.
# Variablerne sorteres efter Gain.

# 14. Cross validation sammenligning -----------------------------------------

# Udtræk af CV-resultater fra logistisk regression (caret-pakken)
log_cv2_results <- log_cv2$results %>%
  select(ROC, Sens, Spec) %>%
  slice(which.max(ROC)) %>%
  mutate(Model = "Logistisk regression")
# log_cv2$results indeholder caret's cross‑validation‑resultater for alle fold.
# Vi vælger kun kolonnerne: ROC, Sens og Spec. 
# slice(which.max(ROC)) vælger den række hvor ROC er højest.
# Vi tilføjer en kolonne med modelnavnet.

# Udtræk af CV-resultater fra Random Forest (caret-pakken)
rf_cv2_results <- rf_cv2$results %>%
  select(ROC, Sens, Spec) %>%
  slice(which.max(ROC)) %>%
  mutate(Model = "Random Forest")
# Der sker det samme som ved logistisk regression, bare for RF. 

# Udtræk af CV-resultater fra XGBoost (xgb.cv)
xgb_cv2_results <- tibble(
  Model = "XGBoost",
  ROC   = max(xgb_cv_m2$evaluation_log$test_auc_mean),
  Sens  = NA,   # xgb.cv giver ikke Sens/Spec direkte
  Spec  = NA
)
# Vi bruger XGBoost's egen indbygget cross-validation.
# XGBoost's cross‑validation (xgb.cv) gemmer resultater i evaluation_log.
# Vi tager den højeste gennemsnitlige test‑AUC.
# XGBoost giver ikke sensitivitet og specificitet i CV, så de sættes til NA.

# Vi samler alle CV-resultater i en tabel
cv_sammenligning2 <- bind_rows(
  log_cv2_results,
  rf_cv2_results,
  xgb_cv2_results
) %>%
  select(Model, ROC, Sens, Spec)
# Vi binder de tre resultattabeller sammen.
# Vi vælger kun de relevante kolonner.
# Resultatet er en pæn sammenligningstabel.

# Vi printer resultatet: 
print(cv_sammenligning2)


# 15. Konklusion ----------------------------------------------------------

# 16. Gem filer til Power BI ----------------------------------------------
write.csv(risk_list_m2, "data/model2_risiko_liste.csv", row.names = FALSE)
write.csv(risk_table_m2, "data/model2_risiko_tabel.csv", row.names = FALSE)
write.csv(economy_table_m2,"data/model2_økonomisk_besparelse.csv", row.names = FALSE)
write.csv(xgb_imp2, "data/model2_xgb_variabel_vigtighed.csv", row.names = FALSE)
write.csv(model_sammenligning2,"data/model2_sammenligning.csv", row.names = FALSE)