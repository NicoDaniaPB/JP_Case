pacman::p_load(tidyverse, cluster, FactoMineR, factoextra, janitor)

# 1. Indlæsning af datasæt ------------------------------------------------
model_data <- readRDS("data/renset_datasæt.rds")

# 2. Indlæsning af konstruerede variabler ---------------------------------
behavior_agg <- readRDS("data/datasæt_konstruerede_variabler.rds")
glimpse(model_data)
# 3. Join af adfærdsvariabler --------------------------------------------
model_data <- model_data %>%
  left_join(behavior_agg, by = "pseudo_id") 
# Vi laver et left_join med model_data og behaviour_agg ved pesudo_id

# 4. Udvælgelse af variabler til clustering -------------------------------
cluster_vars_raw <- model_data %>% 
  select(any_of(c(
    "pseudo_id",
    "koen", "age_group", "age",
    "previous_subscriptions", "previous_campaigns", "previous_trials",
    "continued_after_campaign",
    
    # Adfærd
    "antal_sidevisninger",
    "antal_unikke_sider",
    "andel_restricted",
    "gns_scroll",
    "er_mobil_primær",
    "antal_devices",
    "andel_search",
    "andel_internal",
    "andel_social",
    "andel_indland",
    "andel_kultur",
    "andel_sport",
    "andel_erhverv",  
    "andel_forside",
    "gns_sider_pr_dag"
  )))

# Vi har nu valgt de relevante variabler, som skal indgå i vores klyngeanalyse.
# pseudo_id beholdes til senere join.
# drop_na() fjerner alle ræækker med manglende værdier i nogen af de valgte 
# variabler. Dette sikrer, at distanceberegningen kan køre uden problemer. 

# 5. Konvertering af character til factor ---------------------------------
cluster_vars <- cluster_vars_raw %>% 
  mutate(across(-pseudo_id & where(is.character), as.factor))

# Alle kolonner undtagen pseudo_id forbliver numerisk, hvis den kolonne er 
# numerisk, ellers konverteres den til en factor. 

# 6. Beregning af Gower distance ------------------------------------------
# Vi anvender Gower Distance fordi vi har mixed data. Euklidisk distance som 
# bruges ved PCA, kan kun anvendes til numeriske data. Derfor bruger vi Gower 
# Distance, og fravælger at køre PCA. 
gower_dist <- daisy(cluster_vars %>% select(-pseudo_id), metric = "gower")
# Vi fjerner pseudo_id via select-funktionen, da den ikke skal indgå i distancen.
# dasiy(..., metric = "gower") beregner Gower distance, som kan håndtere mixed
# data (både numerisk og kategoriske), samt normaliserer variabler, så de kan 
# sammenlignes. Resultatet er en distance-matrix, der beskriver hvor lignende
# kunderne er. 

# 7. Hierarkisk clustering -------------------------------------------------
hc <- hclust(gower_dist, method = "ward.D2")
plot(hc, main = "Hierarkisk clustering – kundetyper")
rect.hclust(hc, k = 4, border = "red")
# hclust() laver hierarkisk clustering på distance‑matrixen.
# method = "ward.D2": Ward‑metoden forsøger at minimere variansen 
# inden for klynger. Dette iver ofte kompakte, relativt homogene klynger.
# plot(hc) viser dendrogrammet.
# rect.hclust(..., k = 4) tegner firkanter omkring de 4 klynger, vi vælger 
# visuelt.

# 8. Tilføjelse af klynger ------------------------------------------------

# Vi vælger 4 klynger, fordi dendrogrammet viser et naturligt skæringspunkt her, 
# og fordi fire segmenter giver den mest meningsfulde og forretningsrelevante
# opdeling af abonnenterne.
k <- 4
cluster_vars <- cluster_vars %>% 
  mutate(cluster = factor(cutree(hc, k = k)))
# Vi fastsætter antal klynger til 4.
# cutree(hc, k = 4) skærer dendrogrammet i 4 klynger og giver et klynge‑id til 
# hver observation.
# Vi tilføjer klynge‑id’et som en ny variabel cluster (som factor) til 
# cluster_vars.

# Join tilbage til model_data (uden many-to-many advarsel)
model_data <- model_data %>% 
  left_join(cluster_vars %>% select(pseudo_id, cluster) %>% distinct(), 
            by = "pseudo_id")
# Vi joiner klynge‑label tilbage på det oprindelige model_data via pseudo_id.
# distinct() sikrer, at der kun er én række per pseudo_id i join‑tabellen 
# (for at undgå many‑to‑many advarsler).

# 9. Fjern NA-klyngen -----------------------------------------------------
model_data_clean <- model_data %>% 
  filter(!is.na(cluster))
# Vi fjerner alle rækker, hvor cluster er NA.
# Det kan fx være rækker, der røg ud i drop_na() tidligere og derfor aldrig 
# fik en klynge.

# 10. Clusterprofil --------------------------------------------------------
cluster_profile <- model_data_clean %>% 
  group_by(cluster) %>% 
  summarise(
    across(where(is.numeric), ~ round(mean(.x, na.rm = TRUE), 2)),
    across(where(is.character), ~ names(sort(table(.), decreasing = TRUE))[1]),
    .groups = "drop"
  )
# Vi laver en profil for hver klynge:
# Numeriske variabler: gennemsnit pr. klynge (afrundet til 2 decimaler).
# Character‑variabler: mest hyppige kategori (mode) pr. klynge.
# Resultatet cluster_profile er en kompakt tabel, der beskriver “typisk” adfærd
# og karakteristika for hver klynge.

# Vi ser resultatet
glimpse(cluster_profile)

# 11. Visualisering af klynger --------------------------------------------

# Vi bruger MDS, fordi det er den korrekte metode til at visualisere 
# Gower‑distance ved mixed data, og fordi PCA‑baserede plots ikke kan håndtere
# kategoriske variabler på samme valide måde.

mds <- cmdscale(gower_dist, k = 2, eig = TRUE)
# cmdscale() laver klassisk MDS (Multidimensional Scaling) på distance‑matrixen.
# Vi beder om 2 dimensioner (k = 2), så du kan plotte punkterne i et 2D‑plot.
# MDS forsøger at placere punkterne i et 2D‑rum, så de indbyrdes afstande ligner
# Gower‑distancerne mest muligt.

mds_df <- data.frame(
  Dim1 = mds$points[,1],
  Dim2 = mds$points[,2],
  cluster = cluster_vars$cluster
)
# Vi laver en data frame med:
# Dim1 og Dim2: de to MDS‑koordinater for hver observation.
# cluster: klyngetilhørsforholdet.

# Vi laver plottet:
ggplot(mds_df, aes(Dim1, Dim2, color = cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal() +
  labs(title = "MDS-visualisering af Gower-baserede klynger",
       x = "Dimension 1", y = "Dimension 2")
# Vi har nu et scatterplot, hvor hver observation er et punkt i MDS‑rummet.
# Farven viser klynge.
# Formålet: visuelt at se, hvordan klyngerne ligger i forhold til
# hinanden (overlap, afstand, særskilte grupper).


# Forklaring af klyngerne:

# Klynge 1 – "De tunge, men ustabile brugere"
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

# Visualisering:
# Vi kan se at klynge 2 er mest unik (den gruppe der har størst risiko for churn).
# Vi kan se at klynge 3 og 4 minder mest om hinanden. 
# Klynge 1 er i midten, dvs det er "Mellemgruppen". 
# Man kan også se at gruppen i klynge 1 ikke er tabt, de ligger et mellemsted
# mellem grøn (som er helt tabt) og klynge 3 og 4 som har lavest churn-risiko. 


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
