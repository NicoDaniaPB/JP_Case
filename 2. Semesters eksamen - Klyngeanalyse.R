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
gower_dist <- daisy(cluster_vars %>% select(-pseudo_id), metric = "gower")

# 6B. Elbow-plot og silhouette — begrundelse for valg af k ----------------

# Elbow-metoden: beregn within-cluster sum of squares for k = 2 til 8
# og find det punkt hvor kurven "knækker" — det optimale antal klynger
wss <- map_dbl(2:8, function(k) {
  cutree(hc, k = k) %>%
    { silhouette(., gower_dist) } %>%
    summary() %>%
    { sum((cluster_vars %>% mutate(cl = cutree(hc, k = k)) %>%
             group_by(cl) %>%
             summarise(n = n()))$n) }
})

# Silhouette-scores: måler hvor godt hver observation passer i sin klynge
# Score tæt på 1 = godt placeret, tæt på 0 = på grænsen, negativ = forkert klynge
sil_scores <- map_dbl(2:8, function(k) {
  sil <- silhouette(cutree(hc, k = k), gower_dist)
  mean(sil[, "sil_width"])
})

# Vi plotter silhouette-scores for at bestemme optimalt k
tibble(k = 2:8, silhouette = sil_scores) %>%
  ggplot(aes(x = k, y = silhouette)) +
  geom_line(color = "steelblue", lwd = 1) +
  geom_point(color = "steelblue", size = 3) +
  geom_vline(xintercept = which.max(sil_scores) + 1,
             linetype = "dashed", color = "firebrick") +
  labs(
    title    = "Silhouette-score pr. antal klynger",
    subtitle = "Højere score = bedre intern kohæsion. Rød linje = valgt k",
    x = "Antal klynger (k)", y = "Gennemsnitlig silhouette-score"
  ) +
  theme_minimal()

cat("Optimalt k baseret på silhouette:", which.max(sil_scores) + 1, "\n")



# 7. Hierarkisk clustering -------------------------------------------------
hc <- hclust(gower_dist, method = "ward.D2")
plot(hc, main = "Hierarkisk clustering – kundetyper")
rect.hclust(hc, k = 4, border = "red")

# 8. Tilføjelse af klynger ------------------------------------------------
k <- 4
cluster_vars <- cluster_vars %>% 
  mutate(cluster = factor(cutree(hc, k = k)))

# Join tilbage til model_data
model_data <- model_data %>% 
  left_join(cluster_vars %>% select(pseudo_id, cluster), by = "pseudo_id")

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

# 11. Visualisering af klynger ---------------------------------------------

# Churn-rate pr. klynge — viser hvilke segmenter der er i størst risiko
model_data_clean %>%
  group_by(cluster) %>%
  summarise(
    churn_rate      = mean(continued_after_campaign == 0, na.rm = TRUE),
    hurtig_churn    = mean(fast_churn == 1, na.rm = TRUE),
    antal_kunder    = n()
  ) %>%
  pivot_longer(cols = c(churn_rate, hurtig_churn), names_to = "type", values_to = "andel") %>%
  ggplot(aes(x = cluster, y = andel, fill = type)) +
  geom_col(position = "dodge", alpha = 0.85) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(
    values = c("churn_rate" = "firebrick", "hurtig_churn" = "steelblue"),
    labels = c("Churn ved kampagneslut", "Hurtig churn på listepris")
  ) +
  labs(
    title    = "Churn-rate pr. kundesegment",
    subtitle = "Klynge 2 har højest risiko — Klynge 4 er mest stabil",
    x = "Klynge", y = "Andel", fill = NULL
  ) +
  theme_minimal()

# Aktivitet vs. alder farvet efter klynge — viser segmenternes profil visuelt
model_data_clean %>%
  left_join(behavior_agg %>% select(pseudo_id, antal_sidevisninger), by = "pseudo_id") %>%
  filter(!is.na(antal_sidevisninger), !is.na(age)) %>%
  ggplot(aes(x = age, y = antal_sidevisninger, color = cluster)) +
  geom_point(alpha = 0.5, size = 1.5) +
  scale_color_manual(values = c("1" = "steelblue", "2" = "firebrick",
                                "3" = "darkgreen",  "4" = "darkorange")) +
  labs(
    title    = "Alder vs. aktivitet pr. kundesegment",
    subtitle = "Hvert punkt er en abonnent — farve angiver klyngetilhørsforhold",
    x = "Alder", y = "Antal sidevisninger (30 dage)", color = "Klynge"
  ) +
  theme_minimal()

# Navngivet klyngetabel til rapport
klynge_tabel <- tibble(
  Klynge  = c("1", "2", "3", "4"),
  Navn    = c("De stabile kernebrugere",
              "De lette mobile kampagnebrugere",
              "De engagerede kvalitetslæsere",
              "De ekstremt loyale langtidssubscribers"),
  Risiko  = c("Middel", "Høj", "Lav", "Meget lav"),
  Essens  = c(
    "Høj aktivitet, men lav fastholdelse efter kampagne",
    "Prisfølsom og lavengageret — churner hurtigt",
    "Bred læseadfærd, stabil og værdifuld",
    "Fundamentet i forretningen — ingen churn"
  )
)

print(klynge_tabel)





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
# Essens: En prisfølsom og lavengageret gruppe, der ofte kommer via kampagner og hurtigt falder fra.

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
# Høj aktivitet = lav risiko for churn, og lav aktivitet = hæj risiko for churn.
# Vi kan også se at brugere der læser dybere indhold (artikler, indland, sport,
# analyser) har markant lavere churn end dem der læser noget "overfladisk" son
# fx sport, økonomi og udland. 

