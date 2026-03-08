import random
import pickle
import os
import numpy as np


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
        filename = 'agents/anderson-tiffany-jay-kai/xgb_model'
        with open(filename, "rb") as f:
            self.data  = pickle.load(f)
        self.xgb = self.data['model']
        samples = np.asarray(self.data['sample_covariates'])
        # prices: shape (P,)
        prices = np.arange(0, 150, 0.05)
        # Repeat prices for each sample
        # -> shape becomes (P, N_samples)
        price_grid = np.repeat(prices[:, None], samples.shape[0], axis=1)
        N, C = samples.shape
        P = len(prices)
        # Repeat samples for each price
        # samples -> (N_samples, 3)
        # -> expanded to (P, N_samples, 3)
        # Repeat each sample across all prices -> shape (N, P, C)
        sample_grid = np.repeat(samples[:, None, :], P, axis=1)
        # Repeat prices for each sample -> shape (N, P)
        price_grid = np.tile(prices, (N, 1))
        # Concatenate covariates + price (price last) -> shape (N*P, C+1)
        X = np.concatenate(
            [
                sample_grid.reshape(-1, C),     # (N*P, 3)
                price_grid.reshape(-1, 1)       # (N*P, 1)
            ],
            axis=1
        )
        # Predict probabilities
        probs = self.xgb.predict_proba(X)[:, 1]
        # Reshape back to (N, P)
        probs = probs.reshape(N, P)
        # Expected probability for each price
        self.expected_probs = probs.mean(axis=0)
        self.dp = []
        self.seen_dps = {}
        self.pr_opponent_sell = .5
        ### useful if you want to use a more complex price prediction model
        ### note that you will need to change the name of the path and this agent file when submitting
        ### complications: pickle works with any machine learning models defined in sklearn, xgboost, etc.
        ### however, this does not work with custom defined classes, due to the way pickle serializes objects
        ### refer to './yourteamname/create_model.ipynb' for a quick tutorial on how to use pickle
        # self.filename = './[yourteamname]/trained_model'
        # self.trained_model = pickle.load(open(self.filename, 'rb'))
        ### potentially useful for Part 2 -- When competition is between two agents
        ### and you want to keep track of the opponent's status
        self.opponent_number = 1 - agent_number  # index for opponent
        self.opponent_price_history = []
        self.opponent_model = None
        self.rounds_since_opponent_update = 0
        self.opponent_update_frequency = 100  # how often to update opponent model
        self.opponent_avg_discount = 1.0

        # Adaptive scaling factor
        self.alpha = 0.95  # Start less aggressive to stay closer to optimal
        self.win_count = 0
        self.total_rounds = 0
        self.alpha_min = 0.70  # Minimum discount (raised floor)
        self.alpha_max = 1.00  # Maximum (no discount)
        
        # Initialize covariates tracker
        self.last_customer_covariates = None
        

        

    def demand_distribution(self,prices,covariates,for_filling_dp = False):
        '''
        Array of probabilitities of selling at each price
        At each price:
            Probability of selling:
                = Pr[customer would buy given our price AND opponent price higher than our price] = 
                = Pr[customer would buy given our price] * Pr[opponent price higher than our price || customer buys given our price]
                Initially we may assume that the opponent prices independently since they nothing to base our prices off of
                Overall hard to estimate these probabilities but we can use xgb.predict_proba to find Pr[customer would buy given our price]
            Probability of not selling:
                = Pr[Customer would not buy given our price OR opponent price lower than our price] = 
                = Pr[Customer would not buy given our price] + Pr[opponent price lower than our price] - Pr[Customer would not buy given our price AND opponent price lower than our price]
                And we can further rewrite Pr[Customer would not buy given our price AND opponent price lower than our price]:
                Pr[Customer would not buy given our price] * Pr[opponent price lower than our price || Customer would not buy given our price]
                These conditional probabilities are hard to estimate and dynamically changing

        -------------------------------------------------------------------------------------------------------------------------------------------------
        input: prices: array of prices
               covariates: customer covariates
               for_filling_dp: if this is used to find probabability when filling dp
        output:probability array of selling for each price, probability array of not selling for each price
        '''

        #TODO HOW TO DO THIS??? FOR NOW JUST USE XGB TO PREDICT AND DONT CONSIDER OPPONENT
        # if not for_filling_dp:
        #     covariates = np.asarray(covariates)  # shape (C,)
        #     P = len(prices)
        #     cov_grid = np.tile(covariates, (P, 1))  # shape (P, C)
        #     price_grid = prices.reshape(-1, 1)      # shape (P, 1)
        #     X = np.hstack([cov_grid, price_grid])   # shape (P, C+1)
        #     pr_sell = self.xgb.predict_proba(X)[:, 1]  # shape (P,)
        #     pr_no_sell = 1 - pr_sell

        # else:
        #     pr_sell = self.expected_probs
        #     pr_no_sell = 1 - pr_sell
        # return pr_sell, pr_no_sell

        if not for_filling_dp:
            covariates = np.asarray(covariates)  # shape (C,)
            P = len(prices)
            cov_grid = np.tile(covariates, (P, 1))  # shape (P, C)
            price_grid = prices.reshape(-1, 1)      # shape (P, 1)
            X = np.hstack([cov_grid, price_grid])   # shape (P, C+1)
            
            # Base probability: customer would buy at this price
            pr_buy = self.xgb.predict_proba(X)[:, 1]  # shape (P,)
            
            if self.project_part == 2 and self.last_customer_covariates is not None:
                # Predict opponent's likely price
                opponent_pred_price = self._predict_opponent_price(covariates)
                
                # Probability we undercut opponent
                # Use smooth sigmoid instead of hard threshold for robustness
                price_diff = opponent_pred_price - prices  # Positive when we're cheaper
                pr_undercut = 1.0 / (1.0 + np.exp(-1.0 * price_diff))
                
                # Combined probabilityok: customer buys AND we win
                pr_sell = pr_buy * pr_undercut
            else:
                # Part 1: no competition
                pr_sell = pr_buy
            
            pr_no_sell = 1 - pr_sell
        else:
            # For filling DP: use expected probabilities
            pr_sell = self.expected_probs
            if self.project_part == 2:
                # Use learned competition factor instead of fixed 0.5
                competition_factor = self._estimate_competition_factor()
                pr_sell = pr_sell * competition_factor
            pr_no_sell = 1 - pr_sell

        return pr_sell, pr_no_sell


    def predict_single_step(self, covariates, v_sell, v_no_sell,for_filling_dp = False):
        """
        Find revenue-maximizing price given inputs, using demand_distribution
        to get probability of selling and not selling.
        
        Inputs:
        - covariates: sample covariates
        - v_sell: value if sold
        - v_no_sell: value if not sold
        
        Returns:
        - price_max: revenue-maximizing price
        - revenue_max: maximum revenue
        """
        prices = np.arange(0, 150, 0.05)
        pr_sell, pr_no_sell = self.demand_distribution(prices, covariates,for_filling_dp)
        revenue = pr_sell * (prices + v_sell) + pr_no_sell * v_no_sell
        idx_max = np.argmax(revenue)
        price_max = prices[idx_max]
        revenue_max = revenue[idx_max]
        return price_max, revenue_max

    def generate_dp(self):
        """
        Generate DP table for expected revenue given inventory and replenishment schedule.
        Optimized with caching and float32 precision.
        """
        if self.remaining_inventory in self.seen_dps:
            self.dp = self.seen_dps[self.remaining_inventory]
            return

        # Use float32 for memory efficiency
        self.dp = np.zeros((self.inventory_replenish + 1, self.remaining_inventory + 1), dtype=np.float32)
        dummy_covariates = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Pre-allocate for batch operations
        for time in range(self.inventory_replenish - 1, -1, -1):
            for inventory in range(1, self.remaining_inventory + 1):
                v_sell = self.dp[time + 1, inventory - 1]
                v_no_sell = self.dp[time + 1, inventory]
                _, max_revenue = self.predict_single_step(dummy_covariates, v_sell, v_no_sell, for_filling_dp=True)
                self.dp[time, inventory] = np.float32(max_revenue)
        # Cache as numpy array (faster than converting to list)
        self.seen_dps[self.remaining_inventory] = self.dp.copy()

    def _train_opponent_model(self):
        """Train lightweight model to predict opponent prices."""
        if len(self.opponent_price_history) < 50:
            return
        
        # Extract features and targets
        X = np.array([cov for cov, price in self.opponent_price_history])
        y = np.array([price for cov, price in self.opponent_price_history])
        
        # Use simple linear regression for speed
        from sklearn.linear_model import Ridge
        self.opponent_model = Ridge(alpha=1.0)
        self.opponent_model.fit(X, y)

    def _predict_opponent_price(self, covariates):
        """Predict what opponent would price (without competition)."""
        if self.opponent_model is None or self.project_part == 1:
            # Default assumption: opponent prices around $40 (optimal baseline from data analysis)
            return 40.0

        
        cov_array = np.array(covariates).reshape(1, -1)
        predicted_price = self.opponent_model.predict(cov_array)[0]
        # Clip to reasonable range
        return np.clip(predicted_price, 0, 150)
    
    def _estimate_competition_factor(self):
        """Estimate average probability of winning against opponent."""
        if self.total_rounds < 50 or self.project_part == 1:
            return 0.5  # Default 50% assumption
        
        # Use empirical win rate with some smoothing
        empirical_win_rate = self.win_count / self.total_rounds
        # Smooth towards 0.5 (don't over-commit to recent history)
        return 0.7 * empirical_win_rate + 0.3 * 0.5
        
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
        time_until_replenish = time_until_replenish

        ### TODO - add your code here to potentially update your pricing strategy 
        ### based on what happened in the last round

        if self.project_part == 2 and not np.isnan(last_sale[0]):
            opponent_price = last_sale[1][self.opponent_number]

            if self.last_customer_covariates is not None:
                self.opponent_price_history.append(
                    (self.last_customer_covariates.copy(), opponent_price)
                )
                if len(self.opponent_price_history) > 1000:
                    self.opponent_price_history.pop(0)
            
            self.rounds_since_opponent_update += 1
            if (self.rounds_since_opponent_update >= self.opponent_update_frequency 
                and len(self.opponent_price_history) >= 50):
                self._train_opponent_model()
                self.rounds_since_opponent_update = 0

            # Track opponent's average discount pattern
            if len(self.opponent_price_history) >= 10:
                avg_opponent_price = np.mean([p for c, p in self.opponent_price_history[-50:]])
                # Estimate their effective discount relative to typical optimal prices
                self.opponent_avg_discount = min(avg_opponent_price / 60.0, 1.0)


        # Update alpha based on win/loss
        if self.project_part == 2:
            self.total_rounds += 1
            if did_customer_buy_from_me:
                self.win_count += 1
            
            # Adaptive adjustment based on win rate
            if self.total_rounds % 20 == 0 and self.total_rounds > 0:
                win_rate = self.win_count / self.total_rounds
                # Target ~50-60% win rate (balanced strategy)
                if win_rate < 0.45:  # Losing too much
                    self.alpha = max(self.alpha * 0.93, self.alpha_min)
                elif win_rate > 0.65:  # Winning too much (can price higher)
                    self.alpha = min(self.alpha * 1.07, self.alpha_max)

            # Opponent inventory based adaption -- TODO could be replaced with Nash Equilibrium
            opponent_inventory = inventories[self.opponent_number]
            opponent_inventory_ratio = opponent_inventory / self.inventory_replenish    # normalize opponent inventory

            inventory_alpha_multiplier = 0.8 + 0.4 * opponent_inventory_ratio   # 0.8 to 1.2 multiplier
            self.alpha = self.alpha * inventory_alpha_multiplier


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
        
        # Store covariates BEFORE processing last sale (for opponent model training)
        if self.project_part == 2:
            self.last_customer_covariates = new_buyer_covariates.copy()
        
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)
        # Compute new dp table once in new batch
        if time_until_replenish == self.inventory_replenish:
            self.generate_dp()
        current_inventory = int(inventories[self.this_agent_number])
        
        # Handle zero inventory case
        if current_inventory == 0:
            return 999.0  # Price extremely high when no inventory
        current_time = int(self.inventory_replenish - time_until_replenish)
        # Use numpy array indexing (faster than list indexing)
        expected_revenue_sell = float(self.dp[current_time+1, current_inventory-1])
        expected_revenue_no_sell = float(self.dp[current_time+1, current_inventory])
        # Convert covariates to numpy array for efficiency
        cov_array = np.array(new_buyer_covariates, dtype=np.float32)
        optimal_price, _ = self.predict_single_step(cov_array, expected_revenue_sell, expected_revenue_no_sell)

        
        
        # INVENTORY-AWARE PRICING: Adjust strategy based on inventory pressure
        # Calculate inventory burn rate: how much inventory per time step we should use
        ideal_burn_rate = current_inventory / (time_until_replenish + 1)
        
        # If inventory is critically low, be conservative (raise prices)
        # If inventory is high, be aggressive (lower prices to move inventory)
        if ideal_burn_rate < 0.5:  # Running out of inventory
            inventory_multiplier = 1.08  # Price 8% higher to preserve inventory
        elif ideal_burn_rate > 0.8:  # Too much inventory
            inventory_multiplier = 0.95  # Price lower to move inventory faster
        else:
            inventory_multiplier = 1.0  # Normal pricing
        
        if self.project_part == 2:
            # Have opponent data - be strategic
            opponent_predicted = self._predict_opponent_price(new_buyer_covariates)
            
            # If opponent pricing very high, we can price at optimal
            if opponent_predicted > optimal_price * 1.05:
                return optimal_price * inventory_multiplier
            
            # Use less aggressive alpha to stay closer to optimal
            # Data shows $40 is optimal, we should stay near it
            alpha_based_price = self.alpha * optimal_price
            
            # If opponent is close to optimal already, match them closely
            if abs(opponent_predicted - optimal_price) < 2.0:
                # They're pricing optimally, just slightly undercut
                competitive_price = opponent_predicted * 0.99
            else:
                # Price slightly below opponent prediction (but not too low)
                opponent_undercut_price = opponent_predicted * 0.99
                
                # Choose strategy: undercut opponent but don't go below alpha floor
                competitive_price = max(alpha_based_price, min(opponent_undercut_price, optimal_price))
            
            # Apply inventory adjustment
            competitive_price = competitive_price * inventory_multiplier
            
            # Don't price below $37 (closer to optimal $40)
            competitive_price = np.clip(competitive_price, 37.0, 150.0)
            return competitive_price
        else:
            # Part 1: no competition, just return optimal price
            return optimal_price * inventory_multiplier