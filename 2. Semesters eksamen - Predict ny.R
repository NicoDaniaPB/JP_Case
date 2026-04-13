pacman::p_load(tidyverse, xgboost)

# 1. Indlæs model og klyngecentroider
xgb_model <- read_rds("xgb_model1.rds")
klynge_profiler <- read_rds("klynge_profiler.rds")

# 2. Funktion til at tildele klynge
tildel_klynge <- function(ny_data, klynge_profiler, model_features) {
  
  # Filtrer til model-features
  x <- ny_data %>% select(all_of(model_features)) %>% as.numeric()
  
  # Beregn afstand til hver centroid
  afstande <- apply(klynge_profiler[, model_features], 1, function(centroid) {
    sqrt(sum((x - centroid)^2))
  })
  
  which.min(afstande)
}

# 3. Funktion til at score ny bruger
score_bruger <- function(ny_bruger, xgb_model, klynge_profiler) {
  
  model_features <- xgb_model$feature_names
  
  x_mat <- ny_bruger %>% 
    select(all_of(model_features)) %>% 
    mutate(across(everything(), as.numeric)) %>% 
    data.matrix()
  
  score <- predict(xgb_model, x_mat)
  
  klynge <- tildel_klynge(ny_bruger, klynge_profiler, model_features)
  
  tibble(
    score = score,
    klynge = klynge
  )
}



