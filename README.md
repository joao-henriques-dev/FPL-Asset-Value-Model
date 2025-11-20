# FPL-Asset-Value-Model

The best FPL tool to decide which players to invest our money in. Great for budget management in the beginning of the season and while using a Wildcard.

## Overview

Fantasy Premier League (FPL) is a very popular fantasy football game (11M+ users yearly) where every decision matters. We pick our teams and hope the players deliver. Sometimes we get lucky... sometimes we fail, miserably. And even though we can't take the randomness out of the game, we can minimise it, and data helps us achieve that. So, we gather information about best players and the best teams. We sort the players by xGI or other relevant metric and conclude which players are in-form or "due a haul", then we check the fixtures table and see which teams have the easier runs of games to target them.

All good which this strategy, however, deep down, we're still not considering (and perhaps pondering correctly) all data. FPL assets can return points in a variety of ways, and how exactly do we know if a defender with more defensive contributions is better than a defender with a higher clean sheet potential? How do we know that, just by looking at the isolated numbers? The thing is... we don't, and that's why I decided to build this model.

The FPL Asset Value Model consists of sub-models that:
  - fetch team and player data from .csv files;
  - compute fixture difficulty ratings (FDR) values for all teams given their upcoming games and GW window specified by the user;
  - compute player values based on their position;
  - compute the asset value, combining the previous results (player value and the player's team FDR value) into a final calculation;
  - plot the results in an intuitive manner to determine the best picks for every position based on the given on that GW window.

## Tech Stack

All the codebase is written in *Python* with the help of the following libraries:
  - *Numpy* (for faster computations and input formatting);
  - *Pandas* (for extracting the .csv data);
  - *Scikit-Learn* (for training the Linear Regression sub-model and using it for some of the asset value calculations);
  - *Matplotlib* & *Seaborn* (for showing easy-to-read plots that contain the final results of the model).

## Future Development

I'm looking to integrate this terminal-based program into a web app using basic *HTML*, *CSS* and *JavaScript* code along with the *Flask* framework and a *SQLite* database for now.
I'm aiming for a simple, clean UI that shows the plots for the team and player data and has a section to insert new values or update current ones with an authentication mechanism.
