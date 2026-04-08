pacman::p_load(tidyverse, cluster, FactoMineR, factoextra, janitor)

# 1. Indlæsning af datasæt ------------------------------------------------
model_data <- readRDS("data/renset_datasæt.rds")

# 2. Indlæsning af konstruerede variabler ---------------------------------
behavior_agg <- read_csv("data/datasæt 2 - konstruerede variabler.csv")

# 3. Join af adfærdsvariabler --------------------------------------------
model_data <- model_data %>%
  left_join(behavior_agg, by = "pseudo_id")

# 4. Udvælgelse af variabler til clustering -------------------------------
cluster_vars_raw <- model_data %>% 
  select(
    pseudo_id,
    koen, age_group, age,
    length_group, previous_subscriptions, previous_campaigns, previous_trials,
    fast_churn, continued_after_campaign,
    
    # Konstruerede adfærdsvariabler
    antal_sidevisninger,
    antal_unikke_sider,
    andel_restricted,
    gns_scroll,
    er_mobil_primær,
    antal_devices,
    andel_search,
    andel_internal,
    andel_email,
    andel_social,
    andel_indland,
    andel_udland,
    andel_sport,
    andel_oekonomi,
    andel_forside,
    gns_sider_pr_dag,
    
    type
  ) %>% 
  drop_na()

# 5. Konvertering af character til factor ---------------------------------
cluster_vars <- cluster_vars_raw %>% 
  mutate(across(-pseudo_id, ~ if (is.numeric(.x)) .x else as.factor(.x)))

# 6. Beregning af Gower distance ------------------------------------------
# Vi anvender Gower Distance fordi vi har mixed data. Euklidisk distance som 
# bruges ved PCA, kan kun anvendes til numeriske data. Derfor bruger vi Gower 
# Distance, og fravælger at køre PCA. 
gower_dist <- daisy(cluster_vars %>% select(-pseudo_id), metric = "gower")

# 7. Hierarkisk clustering -------------------------------------------------
hc <- hclust(gower_dist, method = "ward.D2")
plot(hc, main = "Hierarkisk clustering – kundetyper")
rect.hclust(hc, k = 4, border = "red")

# 8. Tilføjelse af klynger ------------------------------------------------

# Vi vælger 4 klynger, fordi dendrogrammet viser et naturligt skæringspunkt her, 
# og fordi fire segmenter giver den mest meningsfulde og forretningsrelevante
# opdeling af abonnenterne.
k <- 4
cluster_vars <- cluster_vars %>% 
  mutate(cluster = factor(cutree(hc, k = k)))

# Join tilbage til model_data (uden many-to-many advarsel)
model_data <- model_data %>% 
  left_join(cluster_vars %>% select(pseudo_id, cluster) %>% distinct(), 
            by = "pseudo_id")

# 9. Fjern NA-klyngen -----------------------------------------------------
model_data_clean <- model_data %>% 
  filter(!is.na(cluster))

# 10. Clusterprofil --------------------------------------------------------
cluster_profile <- model_data_clean %>% 
  group_by(cluster) %>% 
  summarise(
    across(where(is.numeric), ~ round(mean(.x, na.rm = TRUE), 2)),
    across(where(is.character), ~ names(sort(table(.), decreasing = TRUE))[1]),
    .groups = "drop"
  )

glimpse(cluster_profile)

# 11. Visualisering af klynger --------------------------------------------

# Vi bruger MDS, fordi det er den korrekte metode til at visualisere 
# Gower‑distance ved mixed data, og fordi PCA‑baserede plots ikke kan håndtere
# kategoriske variabler på samme valide måde.

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

# Klynge 1 – "De stabile, loyale kernebrugere"
# - Høj aktivitet: mange sidevisninger, mange unikke sider, høj restricted-brug.
# - Primært desktop-brugere.
# - Moderat alder (ca. 60 år).
# - Lav retention efter kampagne og høj churn.
# - Læser primært indland og forside.
# Essens: En tung læsergruppe med højt engagement, men lav fastholdelse.

# Klynge 2 – "De lette, mobile kampagnebrugere"
# - Lav aktivitet: få besøg, få sider, lav scroll.
# - Meget høj mobilandel (70%+).
# - Lav retention og høj churn.
# - Læser primært forside og indland.
# - Kort abonnementslængde.
# Essens: En prisfølsom og lavengageret gruppe, der ofte kommer via kampagner 
# og hurtigt falder fra.

# Klynge 3 – "De engagerede, modne kvalitetslæsere"
# - Høj aktivitet: mange besøg, mange artikler, høj scroll.
# - Blandet device-brug: både mobil og desktop.
# - Høj retention (ca. 68%) og lav churn.
# - Læser bredt: indland, forside, sport, social.
# - Moden alder (ca. 58–59 år).
# Essens: En stærk og stabil kernegruppe med højt engagement og god fastholdelse.

# Klynge 4 – "De ekstremt loyale langtidssubscribers"
# - Meget lang abonnementslængde (over 1.100 dage i gennemsnit).
# - 100% retention efter kampagne.
# - Ingen churn.
# - Høj aktivitet og bred læseadfærd.
# - Blandet device-brug.
# Essens: De mest værdifulde og stabile kunder – fundamentet i forretningen.

# Klynge 2 = Mest risiko for churn, Klynge 4 = Mindst risiko for churn 
# Vi kan også se sammenhæng mellem aktivitet og churn-risiko. 
# Høj aktivitet = lav risiko for churn, og lav aktivitet = høj risiko for churn.
# Vi kan også se at brugere der læser dybere indhold (artikler, indland, sport,
# analyser) har markant lavere churn end dem der læser noget "overfladisk" son
# fx økonomi og udland. 
