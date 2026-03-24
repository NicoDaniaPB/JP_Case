pacman::p_load(tidyverse, readr)

# Vi indlæser dataene
behavior <- read_csv("data/behavior.csv")
cancellation <- read_csv("data/cancellation.csv")
subscription <- read_csv("data/subscription.csv")

view(behavior)
view(cancellation)
view(subscription)

# Vi merger subscription og cancellation
joinet_data <- subscription %>%
  left_join(cancellation, by = "pseudo_id")

# Vi sørger for at der kun er en cancellation pr. kunde
# cancellation_unique <- cancellation %>%
#   group_by(pseudo_id) %>%
#   slice_min(cancellation_date)   

view(joinet_data)
