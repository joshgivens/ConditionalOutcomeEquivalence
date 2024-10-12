library(survival)
library(dplyr)
library(tidyr)
data(colon, package = "survival")


colon_wide <- colon %>%
  mutate(etype = if_else(etype == 1, "Recurrence", "Death")) %>%
  pivot_wider(names_from = etype, values_from = c(status, time)) %>%
  mutate(
    time = pmin(time_Recurrence, time_Death),
    status = pmax(status_Recurrence, status_Death)
  )

write.csv(
  colon_wide,
  "~/OneDrive/Documents/Project/QuantileComparator/RealData/colon.csv"
)
