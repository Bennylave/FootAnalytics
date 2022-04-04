# FootAnalytics

The FootAnalytics project aims to represent event data from football matches. This is done by giving you basic statistics about the players and teams involved (number of shots, pass completion, etc.) and also by plotting in a pitch where certain events took place and their outcomes. This project also contains an xG (expected Goals) model that is a Machine Learning model used to give the probability of goal for any given shot.

## Structure

- Match: this is the main object of the whole project. It contains all the necessary information that will be used to plot the match events and give you statistics.
- utils: In this script you will find some functions that I found useful to when developing class Match. They contain things like a function that can be used to find the match id.
- notebooks: Here you can find notebooks that include:
	- a test notebook where I showcase different methods that you can use with Match to get the information needed for the match that you are searching for.
	-  other xg model related notebooks with an EDA used for the developing the model, the evaluation of this model for the 2015/2016 LaLiga season and a notebook that was used for the model selection.
- constants: a set of data structures containing information that was used throughout the project.
- model_utils: the same concept as utils but applied for the xG model
- requirements.txt: a set of libraries that are required for this project to run

## Data

The data for this project was retrieved from the Statsbomb database. Unfortunately it will not be fully available in this repo as it is too large.

## Additional Notes

This project is still under development. Certain parts of it are still missing (ex: Summary statistics for players and for team) and you may find certain parts of the code inefficient but you can already find the visualization of the events available. The xG model is already developed as well.
