from ast import List
import random
import pickle
import os
# from data.data_helpers_2025 import expected_revenue
import numpy as np
import sklearn
import pandas as pd


'''
This template serves as a starting point for your agent.
'''


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        
        self.project_part = params['project_part'] 

        ### starting remaining inventory and inventory replenish rate are provided
        ## every time the inventory is replenished, it is set to the inventory limit
        ## the inventory_replenish rate is how often the inventory is replenished
        ## for example, we will run with inventory_replenish = 20, with the limit of 11. Then, the inventory will be replenished every 20 time steps (time steps 0, 20, 40, ...) and the inventory will be set to 11 at those time steps. 
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']

        ### useful if you want to use a more complex price prediction model
        ### note that you will need to change the name of the path and this agent file when submitting
        ### complications: pickle works with any machine learning models defined in sklearn, xgboost, etc.
        ### however, this does not work with custom defined classes, due to the way pickle serializes objects
        ### refer to './yourteamname/create_model.ipynb' for a quick tutorial on how to use pickle
        self.filename = 'agents/anderson-tiffany-jay-kai/trained_model'
        self.filename = 'agents/anderson-tiffany-jay-kai/trained_model'
        
        with open(self.filename, 'rb') as f:
            self.model = pickle.load(f)
        
        train_df = pd.read_csv('agents/anderson-tiffany-jay-kai/train_prices_decisions_2025.csv')
        
        # Compute mean of each covariate -- global covariates
        self.mean_covariates = train_df[['Covariate1', 'Covariate2', 'Covariate3']].mean().values

        price_dummy = 50  # TODO change this -- representative price

        # demand distribution based on mean covariates -- global demand distribution
        self.global_demand_dist = self.demand_distribution(self.model, price_dummy, self.mean_covariates)

        self.dp = []
        self.seen_dps = {}
        ### potentially useful for Part 2 -- When competition is between two agents
        ### and you want to keep track of the opponent's status
        # self.opponent_number = 1 - agent_number  # index for opponent

    
    def demand_distribution(self, fitted_model, price, covariates):
        """
        get_prediction_logistic(fitted_model, price, covariates) function from HW3 modified for customer demand distribution
        """
        X_new = pd.DataFrame([{
            "Cov1_high": covariates[0],
            "Cov2_high": covariates[1],
            "Cov3_high": covariates[2],
            "price": price,
        }])

        # Interaction terms
        X_new["price_x_Cov1"] = X_new["price"] * X_new["Cov1_high"]
        X_new["price_x_Cov2"] = X_new["price"] * X_new["Cov2_high"]
        X_new["price_x_Cov3"] = X_new["price"] * X_new["Cov3_high"]

        # Make sure columns match what model expects
        X_new = X_new[fitted_model.feature_names_in_]

        # Return probability of purchase
        return fitted_model.predict_proba(X_new)[:, 1][0]


    def predict_single_step(self,covariates,v_sell,v_no_sell,max_price = 500,price_bucket_size = 5):
        """
        Given demand estimate and expected values at next steps will find optimal price and revenue
        Search on a continuous price range to find the revenue maximizing price given expected.
        This will be used to populate both the expected value dp table and for making real time predictions online
        Inputs:
            covariates: customer covariates
            v_sell: Expected revenue if we sell the product (will be dp[time+1][inventory-1])
            v_no_sell: Expected revenue if don't sell the product(will be dp[time+1][inventory])
            max_price: the upper bound of prices we will consider
            price_bucket_size: how large we want the bins of prices to be(the smaller it is, the more precise but also slower)

        Outputs:
            (best_price, best_revenue)
            best_price will be used in online prediction
            best_revenue will be used to populate the expected value dp table
        """
        #TODO is there better way to find price(can we maybe still use bin search? maybe numerical derivative?)
        best_price = float('inf')
        best_revenue = float('-inf')

        prices = [(i,i+price_bucket_size) for i in range(0,max_price,price_bucket_size)]
        prices_med = [(start + end / 2) for (start, end) in prices] # find midpoint of each bucket -- modeled after HW3

        best_price = float('inf')
        best_revenue = float('-inf')
        for price in prices_med:
            probability_sell = self.demand_distribution(self.model, price, covariates)
            probability_no_sell = 1 -  probability_sell

            expected_value = ((price + v_sell) * probability_sell) + (v_no_sell * probability_no_sell)
            if expected_value > best_revenue:
                best_revenue = expected_value
                best_price = price
        return best_price,best_revenue
    
    def generate_dp(self):
        """At Every n = (self.inventory_replenish) customers generate a new DP table 
           of size (n, self.inventory_limit)
        """  
        self.dp = [[0 for _ in range(self.remaining_inventory + 1)] for _ in range(self.inventory_replenish+1)]

        for time in range(self.inventory_replenish - 1, -1, -1):
            for inventory in range(1, self.remaining_inventory + 1):

                v_sell = self.dp[time + 1][inventory - 1] if (inventory - 1) >= 0 else 0
                v_no_sell = self.dp[time + 1][inventory]

                _, max_revenue = self.predict_single_step(
                    self.mean_covariates,
                    v_sell,
                    v_no_sell
                )

            self.dp[time][inventory] = max_revenue


        return
    
    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        '''
        This function updates your internal state based on the last sale that occurred.
        This template shows you several ways you can keep track of important metrics.
        '''
        ### keep track of who, if anyone, the customer bought from
        did_customer_buy_from_me = (last_sale[0] == self.this_agent_number)
        ### potentially useful for Part 2
        # did_customer_buy_from_opponent = (last_sale[0] == self.opponent_number)

        ### keep track of the prices that were offered in the last sale
        my_last_prices = last_sale[1][self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_last_prices = last_sale[1][self.opponent_number]

        ### keep track of the profit for this agent after the last sale
        my_current_profit = state[self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_current_profit = state[self.opponent_number]

        ### keep track of the inventory levels after the last sale
        self.remaining_inventory = inventories[self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_inventory = inventories[self.opponent_number]

        ### keep track of the time until the next replenishment
        time_until_replenish -= 1

        ### TODO - add your code here to potentially update your pricing strategy 
        ### based on what happened in the last round
        pass

    def action(self, obs):
        '''
        This function is called every time the agent needs to choose an action by the environment.

        The input 'obs' is a 5 tuple, containing the following information:
        -- new_buyer_covariates: a vector of length 3, containing the covariates of the new buyer.
        -- last_sale: a tuple of length 2. The first element is the index of the agent that made the last sale, if it is NaN, then the customer did not make a purchase. The second element is a numpy array of length n_agents, containing the prices that were offered by each agent in the last sale.
        -- state: a vector of length n_agents, containing the current profit of each agent.
        -- inventories: a vector of length n_agents, containing the current inventory level of each agent.
        -- time_until_replenish: an integer indicating the time until the next replenishment, by which time your (and your opponent's, in part 2) remaining inventory will be reset to the inventory limit.

        The expected output is a single number, indicating the price that you would post for the new buyer.
        '''

        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)
        #Compute new dp table once in new batch. This is necessary because base case changes wihn new inventory size
        if time_until_replenish == self.inventory_replenish:
            self.generate_dp()

        current_inventory = inventories[self.this_agent_number]
        current_time = self.inventory_replenish - time_until_replenish
        expected_revenue_sell = self.dp[current_time+1][current_inventory-1]
        expected_revenue_no_sell = self.dp[current_time+1][current_inventory]
        optimal_price,_ = self.predict_single_step(new_buyer_covariates, expected_revenue_sell, expected_revenue_no_sell)

        ### currently output is just a deterministic price for the item
        ### but you are expected to use the new_buyer_covariates
        ### combined with models you come up with using the training data 
        ### and history of prices from each team to set a better price for the item
        return optimal_price

