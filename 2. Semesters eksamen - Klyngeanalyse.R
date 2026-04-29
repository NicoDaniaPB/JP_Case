pacman::p_load(tidyverse, cluster, FactoMineR, factoextra, janitor)

# 1. Indlæsning af datasæt ------------------------------------------------
model_data <- readRDS("data/renset_datasæt.rds")
behavior_agg <- readRDS("data/datasæt_konstruerede_variabler.rds")

# 2. Join af adfærdsvariabler --------------------------------------------
model_data <- model_data %>%
  left_join(behavior_agg, by = "pseudo_id")

# 3. Udvælgelse af variabler til clustering -------------------------------
# Kun input-variabler (ingen churn/retention outcomes)
cluster_vars_raw <- model_data %>% 
  select(any_of(c(
    "pseudo_id",
    # Demografi
    "koen", "age_group", "age",
    # Historik
    "previous_subscriptions", "previous_campaigns", "previous_trials",
    # Adfærd
    "antal_sidevisninger", "antal_unikke_sider", "gns_sider_pr_dag",
    "andel_restricted", "gns_scroll",
    # Platform
    "er_mobil_primær", "antal_devices",
    # Content
    "andel_search", "andel_internal", "andel_social",
    "andel_indland", "andel_kultur", "andel_sport",
    "andel_erhverv", "andel_forside"
  )))

# 4. Konvertering af character til factor ---------------------------------
cluster_vars <- cluster_vars_raw %>% 
  drop_na() %>% 
  mutate(
    across(where(is.character), as.factor),
    across(where(is.logical), as.factor),
    # Transformationer af skæve variabler
    antal_sidevisninger = log1p(antal_sidevisninger),
    gns_sider_pr_dag = log1p(gns_sider_pr_dag)
  )

# 5. Beregning af Gower distance ------------------------------------------
gower_dist <- daisy(cluster_vars %>% select(-pseudo_id), metric = "gower")

# 6. Hierarkisk clustering -------------------------------------------------
hc <- hclust(gower_dist, method = "ward.D2")
plot(hc, main = "Hierarkisk clustering – kundetyper")
rect.hclust(hc, k = 4, border = "red")

# 7. Tilføjelse af klynger ------------------------------------------------
k <- 4
cluster_vars <- cluster_vars %>% 
  mutate(cluster = factor(cutree(hc, k = k)))

# 8. Join tilbage til model_data ------------------------------------------
model_data <- model_data %>% 
  left_join(cluster_vars %>% select(pseudo_id, cluster) %>% distinct(), 
            by = "pseudo_id")

model_data_clean <- model_data %>% filter(!is.na(cluster))

# 9. Clusterprofil --------------------------------------------------------
# OBS vi laver cluster_vars_raw for at forberede de variabler, 
# der skal bruges til selve clustering‑algoritmen.
# Vi bruger model_data_clean til cluster_profile, fordi det er 
# det fulde datasæt med både input‑variabler og alle de andre variabler, 
# vi gerne vil profilere klyngerne på bagefter.
cluster_profile <- model_data_clean %>% 
  group_by(cluster) %>% 
  summarise(
    across(where(is.numeric), ~ round(mean(.x, na.rm = TRUE), 2)),
    across(where(is.character), ~ names(sort(table(.), decreasing = TRUE))[1]),
    .groups = "drop"
  )
print(cluster_profile)
glimpse(cluster_profile)

# 10. Outcome-profilering (kun til analyse, ikke clustering) ---------------
cluster_outcomes <- model_data_clean %>% 
  group_by(cluster) %>% 
  summarise(
    n = n(),
    fast_churn_rate = mean(fast_churn, na.rm = TRUE),
    retention_rate = mean(continued_after_campaign, na.rm = TRUE),
    avg_subscription_length = mean(subscription_length_days, na.rm = TRUE),
    .groups = "drop"
  )
print(cluster_outcomes)
glimpse(cluster_profile)

# 11. Visualisering med MDS -----------------------------------------------
mds <- cmdscale(gower_dist, k = 2, eig = TRUE)
mds_df <- data.frame(
  Dim1 = mds$points[,1],
  Dim2 = mds$points[,2],
  cluster = cluster_vars$cluster
)

ggplot(mds_df, aes(Dim1, Dim2, color = cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal() +
  labs(title = "MDS-visualisering af Gower-baserede klynger",
       x = "Dimension 1", y = "Dimension 2")


# Forklaring af klyngerne:

# Klynge 1 – “De tunge, men ustabile brugere”
# - Høj aktivitet: mange sidevisninger, mange unikke sider og
# høj brug af restricted-indhold.
# - Primært desktop-brugere.
# - Moderat alder (omkring 60 år).
# - Lav retention efter kampagne og relativt høj churn.
# - Læser især indland og forside.
# Essens: En tung læsergruppe med højt engagement, men lav fastholdelse og 
# risiko for frafald.

# Klynge 2 – “De lette, mobile kampagnebrugere”
# - Lav aktivitet: få besøg, få sider og lav scroll.
# - Meget høj mobilandel (70 % eller mere).
# - Lav retention og høj churn.
# - Læser primært forside og indland.
# - Kort abonnementsperiode.
# Essens: En prisfølsom og lavengageret gruppe, der ofte kommer ind via 
# kampagner og hurtigt falder fra igen.

# Klynge 3 – “De engagerede, modne kvalitetslæsere”
# - Høj aktivitet: mange besøg, mange artikler og høj scroll.
# - Blandet device-brug (både mobil og desktop).
# - Høj retention (ca. 68 %) og lav churn.
# - Læser bredt: indland, forside, sport, social m.m.
# - Moden alder (ca. 58–59 år).
# Essens: En stærk og stabil kernegruppe med højt engagement og god fastholdelse.

# Klynge 4 – “De ekstremt loyale langtidssubscribers”
# - Meget lang abonnementsperiode (over 1.100 dage i gennemsnit).
# - 100 % retention efter kampagne.
# - Ingen churn.
# - Høj aktivitet og bred læseadfærd.
# - Blandet device-brug.
# Essens: De mest værdifulde og stabile kunder – fundamentet i forretningen.

# Klynge 2 har den højeste churn-risiko, mens klynge 4 har den laveste.
# Der er en tydelig sammenhæng mellem aktivitet og churn:
# Høj aktivitet = lav churn
# Lav aktivitet = høj churn
# Brugere, der læser dybere og mere redaktionelt indhold 
# (artikler, indland, sport, analyser), har markant lavere churn end dem, 
# der primært læser mere overfladiske kategorier som økonomi og udland.

# 12. Eksportering af datasæt ---------------------------------------------

# Eksportering af datasæt med klynge-profiler
write_csv(cluster_profile, "klynge_profiler.csv")

# Eksportering af datasæt med cluster-labels
write_csv(
  model_data_clean,
  "model_data_med_klynger.csv"
)

# Eksportering af MDS-koordinater
write_csv(
  mds_df,
  "mds_coordinates.csv"
)
