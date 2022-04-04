import pandas as pd
import constants as con
import utils as ut
import matplotsoccer
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
from mplsoccer.pitch import Pitch






class Match:
    """
    The class Match contains all the necessary attributes to generate the plots regarding the Match event.

    competition: the competition name
    season: the season name
    home: home team name
    away: away team name

    """
    def __init__(self, season, competition, home, away):
        self.competition = competition
        self.season = season
        self.home = home
        self.away = away

        # fetching competition and season id
        competition_id, season_id = ut.search_competition(self.competition, self.season)
        # fetching match id
        self.match_id = ut.search_match(competition_id, season_id, self.home, self.away)
        # reading events df from match
        self.events, self.events_players = ut.read_events(self.match_id)

    def get_shots_df(self, name: str, player: bool = True):
        # get match shots as df
        if player:
            events_player_shots = self.events_players[
                (self.events_players['player_name'].str.contains(str(name))) &
                (self.events_players['type_name'] == 'Shot')].copy()
        else:
            events_player_shots = self.events_players[
                (self.events_players['team_name_player'].str.contains(str(name))) &
                (self.events_players['type_name'] == 'Shot')].copy()
        return events_player_shots

    def get_player_fouls_df(self, player_name):
        # get players fouls as three dfs: fouls without cards, fouls with yellow card and fouls with red card

        # Fouls without card
        fouls_no_card = self.events_players[
            (self.events_players['player_name'].str.contains(str(player_name))) &
            (self.events_players['type_name'] == 'Foul Committed') &
            (self.events_players['foul_committed_card_name'].isnull())].copy()
        # Fouls with yellow card
        fouls_yellow_card = self.events_players[
            (self.events_players['player_name'].str.contains(str(player_name))) &
            (self.events_players['type_name'] == 'Foul Committed') &
            (self.events_players['foul_committed_card_name'] == 'Yellow Card')].copy()
        # Fouls with red card
        fouls_red_card = self.events_players[
            (self.events_players['player_name'].str.contains(str(player_name))) &
            (self.events_players['type_name'] == 'Foul Committed') &
            (self.events_players['foul_committed_card_name'].isin(["Red Card", "Second Yellow"]))].copy()

        return fouls_no_card, fouls_yellow_card, fouls_red_card

    def get_player_recoveries_df(self, player_name):
        # get players recoveries as two dfs: successful recoveries and unsuccessful recoveries

        # Fetching all recoveries
        events_player_recoveries = self.events_players[
            (self.events_players['player_name'].str.contains(str(player_name))) &
            (self.events_players['type_name'] == 'Ball Recovery')].copy()

        # Successul recoveries
        events_player_success_recoveries = events_player_recoveries[
            events_player_recoveries["ball_recovery_recovery_failure"].isnull()]

        # Failed recoveries
        events_player_fail_recoveries = events_player_recoveries[
            events_player_recoveries["ball_recovery_recovery_failure"].notnull()]

        return events_player_success_recoveries, events_player_fail_recoveries

    def get_player_clearances_df(self, player_name):
        # Getting players clearances
        events_player_clearances = self.events_players[
            (self.events_players['player_name'].str.contains(str(player_name))) &
            (self.events_players['type_name'] == 'Clearance')].copy()

        # Player clearances with left foot
        events_player_clearances_left = events_player_clearances[
            events_player_clearances["clearance_body_part_name"] == "Left Foot"]
        # Player clearances with right foot
        events_player_clearances_right = events_player_clearances[
            events_player_clearances["clearance_body_part_name"] == "Right Foot"]
        # Player clearances with head
        events_player_clearances_head = events_player_clearances[
            events_player_clearances["clearance_body_part_name"] == "Head"]

        return events_player_clearances_left, events_player_clearances_right, events_player_clearances_head

    def get_player_dispossessions_df(self, player_name):

        # getting player dispossessions
        events_player_dispossessions = self.events_players[
            (self.events_players['player_name'].str.contains(str(player_name))) &
            (self.events_players['type_name'] == 'Dispossessed')].copy()

        return events_player_dispossessions

    def get_player_dribbles_df(self, player_name):

        # Getting player dribbles
        events_player_dribbles = self.events_players[
            (self.events_players['player_name'].str.contains(str(player_name))) &
            (self.events_players['type_name'] == 'Dribble')].copy()

        # Complete dribbles
        events_player_complete_dribble = events_player_dribbles[
            events_player_dribbles["dribble_outcome_name"] == "Complete"]
        # Incomplete dribbles
        events_player_incomplete_dribble = events_player_dribbles[
            events_player_dribbles["dribble_outcome_name"] == "Incomplete"]

        return events_player_complete_dribble, events_player_incomplete_dribble

    def get_player_actions_df(self, player_name):
        # Getting every player's action
        player_actions = self.events_players[
            (self.events_players['player_name'].str.contains(str(player_name))) &
            (self.events_players['team_name_player'] == self.events_players['possession_team_name'])]
        return player_actions

    def get_player_passes_df(self, player_name):

        # Getting passes from a player

        # Reading passes df
        events_passes = ut.read_passes(self.events_players)

        # Filtering by player name
        events_player_pass = events_passes[(events_passes['player_name'].str.contains(str(player_name)))].copy()
        # Getting passes index
        passes_index_list = events_player_pass.index.tolist()
        # Reading passes end location
        events_player_ball_receipt = ut.read_passes_end_location(self.events_players, passes_index_list)

        # Filtering by low and high passes
        events_player_low_pass = events_player_pass[
            events_player_pass['pass_height_name'].isin(['Ground Pass', 'Low Pass'])].copy()
        events_player_high_pass = events_player_pass[events_player_pass['pass_height_name'].isin(['High Pass'])].copy()

        # Filtering by low missed passes
        events_player_low_missed_pass = events_player_low_pass[
            events_player_low_pass['pass_outcome_name'].isin(['Incomplete', 'Out', 'Pass Offside'])].copy()

        # Filtering by low completed passes
        events_player_low_completed_pass = events_player_low_pass[
            events_player_low_pass['pass_outcome_name'].isnull()].copy()

        # Filtering by high missed passes
        events_player_high_missed_pass = events_player_high_pass[
            events_player_high_pass['pass_outcome_name'].isin(['Incomplete', 'Out', 'Pass Offside'])].copy()

        # Filtering by high completed passes
        events_player_high_completed_pass = events_player_high_pass[
            events_player_high_pass['pass_outcome_name'].isnull()].copy()

        return events_player_pass, events_player_low_missed_pass, events_player_low_completed_pass, events_player_high_missed_pass, \
               events_player_high_completed_pass, events_player_ball_receipt

    def get_player_crosses_df(self, player_name):

        # Getting crosses from a player

        # Reading passes df
        events_passes = ut.read_passes(self.events_players)

        # Filtering by passes from a specific player
        events_player_pass = events_passes[(events_passes['player_name'].str.contains(str(player_name)))].copy()

        # Filtering by crosses only
        events_player_crosses = events_player_pass[events_player_pass["pass_cross"].notnull()].copy()

        # Getting indexes of crosses
        passes_index_list = events_player_crosses.index.tolist()

        # Using indexes of crosses for end location
        events_player_cross_ball_receipt = ut.read_passes_end_location(self.events_players, passes_index_list)

        # Filtering by low crosses
        events_player_low_cross = events_player_crosses[
            events_player_crosses['pass_height_name'].isin(['Ground Pass', 'Low Pass'])].copy()

        # Filtering by high crosses
        events_player_high_cross = events_player_crosses[
            events_player_crosses['pass_height_name'].isin(['High Pass'])].copy()

        # Filtering by low missed crosses
        events_player_low_missed_cross = events_player_low_cross[
            events_player_low_cross['pass_outcome_name'].isin(['Incomplete', 'Out', 'Pass Offside'])].copy()

        # Filtering by low completed crosses
        events_player_low_completed_cross = events_player_low_cross[
            events_player_low_cross['pass_outcome_name'].isnull()].copy()

        # Filtering by high missed crosses
        events_player_high_missed_cross = events_player_high_cross[
            events_player_high_cross['pass_outcome_name'].isin(['Incomplete', 'Out', 'Pass Offside'])].copy()

        # Filtering by high completed crosses
        events_player_high_completed_cross = events_player_high_cross[
            events_player_high_cross['pass_outcome_name'].isnull()].copy()

        return events_player_crosses, events_player_low_missed_cross, events_player_low_completed_cross, \
               events_player_high_missed_cross, events_player_high_completed_cross, events_player_cross_ball_receipt

    def get_player_carries_df(self, player_name):

        # Getting player's ball carries

        player_carries = self.events_players[
            (self.events_players['player_name'].str.contains(str(player_name))) &
            (self.events_players['type_name'] == "Carry")]
        return player_carries

    def get_player_fouls_won_df(self, player_name):

        # Getting players fouls won

        player_fouls_won = self.events_players[
            (self.events_players['player_name'].str.contains(str(player_name))) &
            (self.events_players['type_name'] == "Foul Won")]
        return player_fouls_won

    def player_clearances_location(self, player_name):

        # Getting player clearance locations and returning it as dict

        clearances_left, clearances_right, clearances_head = self.get_player_clearances_df(player_name)

        x_left = clearances_left.location_x.to_list()
        y_left = clearances_left.location_y.to_list()

        x_right = clearances_right.location_x.to_list()
        y_right = clearances_right.location_y.to_list()

        x_head = clearances_head.location_x.to_list()
        y_head = clearances_head.location_y.to_list()

        clearances_dict = {
            "x_left": x_left,
            "y_left": y_left,
            "x_right": x_right,
            "y_right": y_right,
            "x_head": x_head,
            "y_head": y_head
        }

        return clearances_dict

    def player_fouls_won_locations(self, player_name):

        # Getting player fouls locations and returning it as dict

        player_fouls_won = self.get_player_fouls_won_df(player_name)

        x_foul_won = player_fouls_won.location_x.to_list()
        y_foul_won = player_fouls_won.location_y.to_list()

        return x_foul_won, y_foul_won

    def player_carry_locations(self, player_name):

        # Getting player carries locations adn end locations and returning it as dict

        player_carries = self.get_player_carries_df(player_name)

        x_carry = player_carries.location_x.to_list()
        y_carry = player_carries.location_y.to_list()

        end_locations = player_carries.carry_end_location.to_list()

        x_end_location = [i[0] for i in end_locations]
        y_end_location = [i[1] for i in end_locations]

        fouls_dict = {
            "x_carry": x_carry,
            "y_carry": y_carry,
            "x_end_location": x_end_location,
            "y_end_location": y_end_location
        }

        return fouls_dict

    def player_fouls_locations(self, player_name):

        # Getting player fouls locations for every type of foul and returning it as dict

        fouls_no_card, fouls_yellow_card, fouls_red_card = self.get_player_fouls_df(player_name)

        x_fouls_no_card = fouls_no_card.location_x.to_list()
        y_fouls_no_card = fouls_no_card.location_y.to_list()

        x_fouls_yellow_card = fouls_yellow_card.location_x.to_list()
        y_fouls_yellow_card = fouls_yellow_card.location_y.to_list()

        x_fouls_red_card = fouls_red_card.location_x.to_list()
        y_fouls_red_card = fouls_red_card.location_y.to_list()

        fouls_dict = {
            "x_fouls_no_card": x_fouls_no_card,
            "y_fouls_no_card": y_fouls_no_card,
            "x_fouls_yellow_card": x_fouls_yellow_card,
            "y_fouls_yellow_card": y_fouls_yellow_card,
            "x_fouls_red_card": x_fouls_red_card,
            "y_fouls_red_card": y_fouls_red_card
        }

        return fouls_dict

    def player_recoveries_locations(self, player_name):

        # Getting player recoveries locations for every type of foul and returning it as dict

        success_recoveries, fail_recoveries = self.get_player_recoveries_df(player_name)

        x_location_success_recoveries = success_recoveries.location_x.to_list()
        y_location_success_recoveries = success_recoveries.location_y.to_list()

        x_location_fail_recoveries = fail_recoveries.location_x.to_list()
        y_location_fail_recoveries = fail_recoveries.location_y.to_list()

        recoveries_dict = {
            "x_success": x_location_success_recoveries,
            "y_success": y_location_success_recoveries,
            "x_fail": x_location_fail_recoveries,
            "y_fail": y_location_fail_recoveries,
        }

        return recoveries_dict

    def player_dispossessions_locations(self, player_name):

        # Getting player dispossessions and returning it as dict

        df = self.get_player_dispossessions_df(player_name)

        x_locations_dispossessions = df.location_x.to_list()
        y_locations_dispossessions = df.location_y.to_list()

        return x_locations_dispossessions, y_locations_dispossessions

    def shots_locations(self, name, player=True):
        if player:
            df = self.get_shots_df(name, player=True)
        else:
            df = self.get_shots_df(name, player=False)

        df_dict = ut.split_by_event_chars(df, ["shot_outcome_name", "shot_body_part_name"])

        df_shot_location_dict = {k: [v.location_x.tolist(), v.location_y.tolist()] for k, v in df_dict.items()}
        return df_shot_location_dict

    def player_pass_locations(self, player_name):
        df_pass, df_miss_low, df_complete_low, df_miss_high, df_complete_high, df_ball_receipt = self.get_player_passes_df(
            player_name)

        x_passes = df_pass.location_x.to_list()
        y_passes = df_pass.location_y.to_list()

        x_location_miss_low_passes = df_miss_low.location_x.to_list()
        y_location_miss_low_passes = df_miss_low.location_y.to_list()

        x_location_complete_low_passes = df_complete_low.location_x.to_list()
        y_location_complete_low_passes = df_complete_low.location_y.to_list()

        x_location_miss_high_passes = df_miss_high.location_x.to_list()
        y_location_miss_high_passes = df_miss_high.location_y.to_list()

        x_location_complete_high_passes = df_complete_high.location_x.to_list()
        y_location_complete_high_passes = df_complete_high.location_y.to_list()

        x_location_ball_receipt = df_ball_receipt.location_x.to_list()
        y_location_ball_receipt = df_ball_receipt.location_y.to_list()

        player_pass_dict = {
            "x_passes": x_passes,
            "y_passes": y_passes,
            "x_miss_low": x_location_miss_low_passes,
            "y_miss_low": y_location_miss_low_passes,
            "x_complete_low": x_location_complete_low_passes,
            "y_complete_low": y_location_complete_low_passes,
            "x_miss_high": x_location_miss_high_passes,
            "y_miss_high": y_location_miss_high_passes,
            "x_complete_high": x_location_complete_high_passes,
            "y_complete_high": y_location_complete_high_passes,
            "x_end_location": x_location_ball_receipt,
            "y_end_location": y_location_ball_receipt
        }

        return player_pass_dict

    def player_cross_locations(self, player_name):
        df_cross, df_miss_low, df_complete_low, df_miss_high, \
        df_complete_high, df_ball_receipt = self.get_player_crosses_df(
            player_name)

        x_cross = df_cross.location_x.to_list()
        y_cross = df_cross.location_y.to_list()

        x_miss_low = df_miss_low.location_x.to_list()
        y_miss_low = df_miss_low.location_y.to_list()

        x_complete_low = df_complete_low.location_x.to_list()
        y_complete_low = df_complete_low.location_y.to_list()

        x_miss_high = df_miss_high.location_x.to_list()
        y_miss_high = df_miss_high.location_y.to_list()

        x_complete_high = df_complete_high.location_x.to_list()
        y_complete_high = df_complete_high.location_y.to_list()

        x_cross_end_location = df_ball_receipt.location_x.to_list()
        y_cross_end_location = df_ball_receipt.location_y.to_list()

        player_cross_dict = {
            "x_cross": x_cross,
            "y_cross": y_cross,
            "x_miss_low": x_miss_low,
            "y_miss_low": y_miss_low,
            "x_complete_low": x_complete_low,
            "y_complete_low": y_complete_low,
            "x_miss_high": x_miss_high,
            "y_miss_high": y_miss_high,
            "x_complete_high": x_complete_high,
            "y_complete_high": y_complete_high,
            "x_cross_end_location": x_cross_end_location,
            "y_cross_end_location": y_cross_end_location
        }

        return player_cross_dict

    def player_dribbles_locations(self, player_name):
        success_dribbles, fail_dribbles = self.get_player_dribbles_df(player_name)

        x_location_success_dribbles = success_dribbles.location_x.to_list()
        y_location_success_dribbles = success_dribbles.location_y.to_list()

        x_location_fail_dribbles = fail_dribbles.location_x.to_list()
        y_location_fail_dribbles = fail_dribbles.location_y.to_list()

        dribbles_dict = {
            "x_success": x_location_success_dribbles,
            "y_success": y_location_success_dribbles,
            "x_fail": x_location_fail_dribbles,
            "y_fail": y_location_fail_dribbles
        }

        return dribbles_dict

    def player_actions_locations(self, player_name):
        player_actions = self.get_player_actions_df(player_name)

        player_actions_dict = {
            "x": player_actions["location_x"].to_list(),
            "y": player_actions["location_y"].to_list()
        }

        return player_actions_dict

    def test_player_carries(self, player_name):

        event_name = "Carry"

        carries_dict = ut.get_event_df(df=self.events_players, player_name=player_name, event_name=event_name)
        carries_location_dict = ut.get_event_locations(carries_dict, end_location=True, event_name=event_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.arrows(carries_location_dict["X_Carry"],
                     carries_location_dict["Y_Carry"],
                     carries_location_dict["X_Carry_End_Location"],
                     carries_location_dict["Y_Carry_End_Location"],
                     ax=axs, color="white", width=2, alpha=0.75, linestyle="--")

        plt.show()

    def player_carries(self, player_name):

        carries_dict = self.player_carry_locations(player_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.arrows(carries_dict["x_carry"], carries_dict["y_carry"], carries_dict["x_end_location"],
                     carries_dict["y_end_location"],
                     ax=axs, color="white", width=2, alpha=0.75, linestyle="--")

        plt.show()

    def test_player_clearances(self, player_name):

        event_name = "Clearance"
        event_values = {
            "left": "Left Foot",
            "right": "Right Foot",
            "head": "Head"
        }

        clearances_dict = ut.get_event_df(df=self.events_players, player_name=player_name, event_name=event_name,
                                          event_values=event_values, event_values_col="clearance_body_part_name")
        clearances_location_dict = ut.get_event_locations(event_dict=clearances_dict, event_name=event_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(clearances_location_dict["X_left"], clearances_location_dict["Y_left"], s=150,
                      edgecolors="white",
                      c="blue", marker="<", ax=axs, label="Left Foot")

        pitch.scatter(clearances_location_dict["X_right"], clearances_location_dict["Y_right"],
                      s=150, edgecolors="white",
                      c="blue", marker=">", ax=axs, label="Right Foot")

        pitch.scatter(clearances_location_dict["X_head"], clearances_location_dict["Y_head"], s=150,
                      edgecolors="white",
                      c="blue", marker="^", ax=axs, label="Head")

        plt.legend(loc=1)
        plt.show()

    def player_clearances(self, player_name):
        clearances_dict = self.player_clearances_location(player_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(clearances_dict["x_left"], clearances_dict["y_left"], s=150, edgecolors="white",
                      c="blue", marker="<", ax=axs, label="Left Foot")

        pitch.scatter(clearances_dict["x_right"], clearances_dict["y_right"], s=150, edgecolors="white",
                      c="blue", marker=">", ax=axs, label="Right Foot")

        pitch.scatter(clearances_dict["x_head"], clearances_dict["y_head"], s=150, edgecolors="white",
                      c="blue", marker="^", ax=axs, label="Head")

        plt.legend(loc=1)
        plt.show()

    def player_fouls(self, player_name):
        fouls_dict = self.player_fouls_locations(player_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(fouls_dict["x_fouls_no_card"], fouls_dict["y_fouls_no_card"], s=150, edgecolors="black",
                      c="white", marker="o", ax=axs, label="No card")

        pitch.scatter(fouls_dict["x_fouls_yellow_card"], fouls_dict["y_fouls_yellow_card"], s=150, edgecolors="black",
                      c="gold", marker="o", ax=axs, label="Yellow card")

        pitch.scatter(fouls_dict["x_fouls_red_card"], fouls_dict["y_fouls_red_card"], s=150, edgecolors="black",
                      c="red", marker="o", ax=axs, label="Red card")

        plt.legend(loc=1)
        plt.show()

    def test_player_fouls(self, player_name):
        event_name = "Foul Committed"
        event_values = {
            "no_card": "Null",
            "yellow": "Yellow Card",
            "red": ["Second Yellow", "Red Card"]
        }

        fouls_dict = ut.get_event_df(df=self.events_players, player_name=player_name, event_name=event_name,
                                     event_values=event_values, event_values_col="foul_committed_card_name")
        fouls_location_dict = ut.get_event_locations(event_dict=fouls_dict, event_name=event_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(fouls_location_dict["X_no_card"], fouls_location_dict["Y_no_card"], s=150,
                      edgecolors="black",
                      c="white", marker="o", ax=axs, label="No Card")

        pitch.scatter(fouls_location_dict["X_yellow"], fouls_location_dict["Y_yellow"],
                      s=150, edgecolors="black",
                      c="gold", marker="o", ax=axs, label="Yellow Card")

        pitch.scatter(fouls_location_dict["X_red"], fouls_location_dict["Y_red"], s=150,
                      edgecolors="black",
                      c="red", marker="o", ax=axs, label="Red Card")

        plt.legend(loc=1)
        plt.show()

    def player_dispossessions(self, player_name):
        x, y = self.player_dispossessions_locations(player_name)
        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(x, y, s=150, edgecolors="white",
                      c="red", marker="o", ax=axs)
        plt.show()

    def test_player_dispossessions(self, player_name):
        event_name = "Dispossessed"

        dispossession_dict = ut.get_event_df(df=self.events_players, player_name=player_name, event_name=event_name)
        dispossession_location_dict = ut.get_event_locations(event_dict=dispossession_dict, event_name=event_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(dispossession_location_dict["X_Dispossessed"], dispossession_location_dict["Y_Dispossessed"],
                      s=150, edgecolors="white",
                      c="red", marker="o", ax=axs)
        plt.show()

    def player_fouls_won(self, player_name):
        x, y = self.player_fouls_won_locations(player_name)
        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(x, y, s=150, edgecolors="white",
                      c="blue", marker="o", ax=axs)
        plt.show()

    def test_player_fouls_won(self, player_name):
        event_name = "Foul Won"

        dispossession_dict = ut.get_event_df(df=self.events_players, player_name=player_name, event_name=event_name)
        dispossession_location_dict = ut.get_event_locations(event_dict=dispossession_dict, event_name=event_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(dispossession_location_dict["X_Foul_Won"], dispossession_location_dict["Y_Foul_Won"], s=150,
                      edgecolors="white",
                      c="blue", marker="o", ax=axs)
        plt.show()

    def player_shots(self, name):
        df_shot_location_dict = self.shots_locations(name, player=True)
        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        for k, v in df_shot_location_dict.items():
            outcome = k.split("_")[0]
            body_part = k.split("_")[1]

            pitch.scatter(v[0], v[1], s=150, edgecolors="black", label=k.replace("_", " "),
                          c=con.shots_colors_dict[outcome], marker=con.shots_markers_dict[body_part], ax=axs)

        plt.legend(con.shots_tuple_legend,
                   con.shots_tuple_labels, numpoints=1,
                   loc=1)
        plt.show()

    def test_player_shots(self, name):
        event_name = "Shot"

        shot_dict = ut.get_event_df(self.events_players, player_name=name, event_name=event_name)

        shot_dict = ut.split_by_event_chars(shot_dict["Shot"], ["shot_outcome_name", "shot_body_part_name"])

        shot_location_dict = {k: [v.location_x.tolist(), v.location_y.tolist()] for k, v in shot_dict.items()}

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        for k, v in shot_location_dict.items():
            outcome = k.split("_")[0]
            body_part = k.split("_")[1]

            pitch.scatter(v[0], v[1], s=150, edgecolors="black", label=k.replace("_", " "),
                          c=con.shots_colors_dict[outcome], marker=con.shots_markers_dict[body_part], ax=axs)

        plt.legend(con.shots_tuple_legend,
                   con.shots_tuple_labels, numpoints=1,
                   loc=1)
        plt.show()

    def team_shots(self, name):
        df_shot_location_dict = self.shots_locations(name, player=False)
        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        for k, v in df_shot_location_dict.items():
            outcome = k.split("_")[0]
            body_part = k.split("_")[1]

            pitch.scatter(v[0], v[1], s=150, edgecolors="black", label=k.replace("_", " "),
                          c=con.shots_colors_dict[outcome], marker=con.shots_markers_dict[body_part], ax=axs)

        plt.legend(con.shots_tuple_legend,
                   con.shots_tuple_labels, numpoints=1,
                   loc=8)
        plt.show()

    def player_recoveries(self, player_name):
        recoveries = self.player_recoveries_locations(player_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(recoveries["x_success"], recoveries["y_success"], s=150, edgecolors="white", color="blue",
                      marker="^",
                      label="Successful Recovery", ax=axs)
        pitch.scatter(recoveries["x_fail"], recoveries["y_fail"], s=150, edgecolors="white", color="red", marker="^",
                      label="Failed Recovery", ax=axs)
        plt.legend(loc=8)
        plt.show()


    def test_player_recoveries(self, player_name):
        event_name = "Ball Recovery"
        event_values = {
            "success": "Null",
            "failure": "Not Null"
        }

        recoveries_dict = ut.get_event_df(df=self.events_players, player_name=player_name, event_name=event_name,
                                     event_values=event_values, event_values_col="ball_recovery_recovery_failure")
        recoveries_location_dict = ut.get_event_locations(event_dict=recoveries_dict, event_name=event_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(recoveries_location_dict["X_success"], recoveries_location_dict["Y_success"], s=150,
                      edgecolors="black",
                      c="blue", marker="o", ax=axs, label="Complete")

        pitch.scatter(recoveries_location_dict["X_failure"], recoveries_location_dict["Y_failure"], s=150,
                      edgecolors="black",
                      c="red", marker="o", ax=axs, label="Incomplete")

        plt.legend(loc=1)
        plt.show()


    def player_dribbles(self, player_name):
        dribbles_dict = self.player_dribbles_locations(player_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(dribbles_dict["x_success"], dribbles_dict["y_success"], s=150, edgecolors="white", color="blue",
                      marker="^",
                      label="Complete", ax=axs)
        pitch.scatter(dribbles_dict["x_fail"], dribbles_dict["y_fail"], s=150, edgecolors="white", color="red",
                      marker="^",
                      label="Incomplete", ax=axs)
        plt.legend(loc=8)
        plt.show()


    def test_player_dribbles(self, player_name):
        event_name = "Dribble"
        event_values = {
            "success": "Complete",
            "failure": "Incomplete"
        }

        dribbles_dict = ut.get_event_df(df=self.events_players, player_name=player_name, event_name=event_name,
                                     event_values=event_values, event_values_col="dribble_outcome_name")
        dribbles_location_dict = ut.get_event_locations(event_dict=dribbles_dict, event_name=event_name)

        pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
        fig, axs = pitch.draw(figsize=(12, 10))

        pitch.scatter(dribbles_location_dict["X_success"], dribbles_location_dict["Y_success"], s=150,
                      edgecolors="black",
                      c="blue", marker="o", ax=axs, label="Complete")

        pitch.scatter(dribbles_location_dict["X_failure"], dribbles_location_dict["Y_failure"], s=150,
                      edgecolors="black",
                      c="red", marker="o", ax=axs, label="Incomplete")

        plt.legend(loc=1)
        plt.show()

    def player_passes(self, player_name, arrows=True):

        pass_dict = self.player_pass_locations(player_name)

        if arrows:
            pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
            fig, axs = pitch.draw(figsize=(12, 10))

            pitch.scatter(pass_dict["x_miss_high"], pass_dict["y_miss_high"], s=150, marker='d', color='red',
                          edgecolors="white",
                          label="Missed High", ax=axs)
            pitch.scatter(pass_dict["x_miss_low"], pass_dict["y_miss_low"], s=150, color='red', edgecolors="white",
                          label="Missed Low", ax=axs)
            pitch.scatter(pass_dict["x_complete_low"], pass_dict["y_complete_low"], s=150, color='blue',
                          edgecolors="white", label="Complete Low",
                          ax=axs)
            pitch.scatter(pass_dict["x_complete_high"], pass_dict["y_complete_high"], s=150, marker="d", color='blue',
                          edgecolors="white",
                          label="Complete High", ax=axs)

            pitch.arrows(pass_dict["x_passes"], pass_dict["y_passes"], pass_dict["x_end_location"],
                         pass_dict["y_end_location"],
                         ax=axs, color="white", width=1, alpha=0.5, linestyle="--")

            plt.legend(loc=8)
            plt.title(str(player_name) + "´s passes")
            plt.show()

        else:
            pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
            fig, axs = pitch.draw(figsize=(12, 10))

            pitch.scatter(pass_dict["x_miss_high"], pass_dict["y_miss_high"], s=150, marker='d', color='red',
                          edgecolors="white",
                          label="Missed High", ax=axs)
            pitch.scatter(pass_dict["x_miss_low"], pass_dict["y_miss_low"], s=150, color='red', edgecolors="white",
                          label="Missed Low", ax=axs)
            pitch.scatter(pass_dict["x_complete_low"], pass_dict["y_complete_low"], s=150, color='blue',
                          edgecolors="white", label="Complete Low",
                          ax=axs)
            pitch.scatter(pass_dict["x_complete_high"], pass_dict["y_complete_high"], s=150, marker="d", color='blue',
                          edgecolors="white",
                          label="Complete High", ax=axs)
            plt.legend(loc=8)
            plt.title(str(player_name) + "´s passes")
            plt.show()

            pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
            fig, axs = pitch.draw(figsize=(12, 10))
            pitch.scatter(pass_dict["x_end_location"], pass_dict["y_end_location"], s=150, marker='8', color='white',
                          edgecolors="black", ax=axs)
            plt.title(str(player_name) + "´s passes end locations")
            plt.show()

    def player_crosses(self, player_name, arrows=True):

        cross_dict = self.player_cross_locations(player_name)

        if arrows:
            pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
            fig, axs = pitch.draw(figsize=(12, 10))

            pitch.scatter(cross_dict["x_miss_high"], cross_dict["y_miss_high"], s=150, marker='d', color='red',
                          edgecolors="white",
                          label="Missed High", ax=axs)
            pitch.scatter(cross_dict["x_miss_low"], cross_dict["y_miss_low"], s=150, color='red', edgecolors="white",
                          label="Missed Low", ax=axs)
            pitch.scatter(cross_dict["x_complete_low"], cross_dict["y_complete_low"], s=150, color='blue',
                          edgecolors="white", label="Complete Low",
                          ax=axs)
            pitch.scatter(cross_dict["x_complete_high"], cross_dict["y_complete_high"], s=150, marker="d", color='blue',
                          edgecolors="white",
                          label="Complete High", ax=axs)

            pitch.arrows(cross_dict["x_cross"], cross_dict["y_cross"], cross_dict["x_cross_end_location"],
                         cross_dict["y_cross_end_location"],
                         ax=axs, color="white", width=1, alpha=0.5, linestyle="--")

            plt.legend(loc=8)
            plt.title(str(player_name) + "´s crosses")
            plt.show()

        else:
            pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
            fig, axs = pitch.draw(figsize=(12, 10))

            pitch.scatter(cross_dict["x_miss_high"], cross_dict["y_miss_high"], s=150, marker='d', color='red',
                          edgecolors="white",
                          label="Missed High", ax=axs)
            pitch.scatter(cross_dict["x_miss_low"], cross_dict["y_miss_low"], s=150, color='red', edgecolors="white",
                          label="Missed Low", ax=axs)
            pitch.scatter(cross_dict["x_complete_low"], cross_dict["y_complete_low"], s=150, color='blue',
                          edgecolors="white", label="Complete Low",
                          ax=axs)
            pitch.scatter(cross_dict["x_complete_high"], cross_dict["y_complete_high"], s=150, marker="d", color='blue',
                          edgecolors="white",
                          label="Complete High", ax=axs)
            plt.legend(loc=8)
            plt.title(str(player_name) + "´s passes")
            plt.show()

            pitch = Pitch(line_zorder=2, figsize=(4.4, 6.4), pitch_color='green')
            fig, axs = pitch.draw(figsize=(12, 10))
            pitch.scatter(cross_dict["x_cross_end_location"], cross_dict["y_cross_end_location"], s=150, marker='8',
                          color='white',
                          edgecolors="black", ax=axs)
            plt.title(str(player_name) + "´s passes end locations")
            plt.show()

    def player_xg(self, player_name):

        player_shots = self.get_shots_df(player_name)
        shots_xg = pd.read_csv("data/clean/xg/predictions_" + self.season.replace("/", "_") + ".csv")

        player_shots_xg = player_shots.join(shots_xg.set_index("id"), on="id", how="inner")
        player_shots_xg = player_shots_xg[["pred", "shot_outcome_name", "minute"]].copy()

        player_shots_xg["pred_cumsum"] = player_shots_xg["pred"].cumsum()

        print(player_shots_xg)
        player_shots_xg.loc[0] = [0, np.nan, 0, 0]
        player_shots_xg.loc[player_shots_xg.index.max() + 1] = [0, np.nan, self.events_players["minute"].max(),
                                                                player_shots_xg["pred_cumsum"].max()]
        player_shots_xg = player_shots_xg.sort_index(ascending=True)

        plt.step(player_shots_xg["minute"], player_shots_xg["pred_cumsum"], where="post")
        plt.show()

    def goal_possessions(self):
        events_goal_conceded = self.events[self.events.goalkeeper_type_name == 'Goal Conceded']
        events_goal_conceded.reset_index(inplace=True)

        possessions_goals_conceded = list()
        for i in range(events_goal_conceded.shape[0]):
            possessions_goals_conceded.append(events_goal_conceded.possession[i])

        return possessions_goals_conceded

    def show_goals(self, figsize=16):
        goals_possessions = self.goal_possessions()

        events_goals = self.events_players[self.events_players['type_name'].isin(
            ['Pass', 'Carry', 'Shot', 'Interception', 'Block', 'Ball Recovery'])].copy()
        events_goals['type_name'] = events_goals.copy()['type_name'].copy().replace('Carry', 'Dribble')

        events_goals['location_x'] = events_goals['location_x'].copy() * 105 / 120
        events_goals['location_y'] = abs((events_goals.copy()['location_y'].copy() * 68 / 80) - 68)

        events_goals["location_x"] = np.where(events_goals['possession_team_name'] != events_goals['team_name_player'],
                                              105 - events_goals["location_x"],
                                              events_goals["location_x"])

        events_goals["location_y"] = np.where(events_goals['possession_team_name'] != events_goals['team_name_player'],
                                              68 - events_goals["location_y"],
                                              events_goals["location_y"])

        events_goals = events_goals.sort_index()

        for i in goals_possessions:
            if events_goals[events_goals['possession'] == i].tail(10).empty:
                goal = events_goals[events_goals['possession'] == i - 1].tail(10)
            else:
                goal = events_goals[events_goals['possession'] == i].tail(10)

            matplotsoccer.actions(
                color='green',
                location=goal[["location_x", "location_y"]],
                action_type=goal.type_name,
                label=goal[["minute", "second", "type_name", "player_name", "team_name_player"]],
                labeltitle=["minute", "second", "actiontype", "player", "team"],
                zoom=True,
                figsize=figsize)

    def player_actions(self, player_name):
        player_actions_locations_dict = self.player_actions_locations(player_name)

        x, y = ut.convert_xy_locations(player_actions_locations_dict["x"], player_actions_locations_dict["y"])

        pitch = Pitch(line_zorder=2, pitch_color='gray')

        fig, ax = pitch.draw(figsize=(12, 10))

        pitch.kdeplot(x, y,
                      ax=ax,
                      levels=100,
                      cut=12,
                      shade=True,
                      cmap=cmr.pride)

        plt.show()

    def teams_stats(self):
        return None
