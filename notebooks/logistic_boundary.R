
library("tidyverse")
library("viridis")
sigmoid <- function(pi) {
  1 / (1 + exp(-pi))
}

generate_probs <- function(sigma=2) {
  theta <- rnorm(3, 0, sigma)
  u <- seq(-1, 1, 0.01)
  x <- list()
  k <- 1
  for (i in seq_along(u)) {
    for (j in seq_along(u)) {
      x[[k]] <- c(u[i], u[j])
      k <- k + 1
    }
  }

  x <- do.call(rbind, x) %>%
    as_data_frame()
  x$intercept <- 1
  x$prob <- sigmoid(t(theta %*% t(x)))
  x
}

plot_probs <- function(x) {
  ggplot(x) +
    geom_tile(
      aes(
        x = V1,
        y = V2,
        fill = prob
      )
    ) +
    geom_vline(xintercept = 0) +
    geom_hline(yintercept = 0) +
    scale_x_continuous("x[1]", expand = c(0, 0)) +
    scale_y_continuous("x[2]", expand = c(0, 0)) +
    scale_fill_viridis() +
    theme(legend.position="none")
}

for (i in seq_len(10)) {
  x <- generate_probs()
  p <- plot_probs(x)
  ggsave(sprintf("sigmoid_plot_%s.png", i))
}
