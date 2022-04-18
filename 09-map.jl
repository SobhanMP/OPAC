using Revise
using Opaac
lg = mk_land_graph(21, 21; cap_range=1:5, cost_range=1:5)
ag, a2l, l2a = mk_air_graph(21, 21, 5)

mkpath("images")

draw_plot(lg; fn="images/01-lg.pdf")
draw_plot(ag; fn="images/02-ag.pdf")