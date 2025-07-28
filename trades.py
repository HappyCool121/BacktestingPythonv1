# trades module, which contains trade class, allowing us to create trades objects to be used in the backtest.
import pandas as pd

# --- Module 4: Trades object ---
class Trades:
    """
    Trades class used to create new trades objects, with parameters that can be changed any time.
    To create a trade, a dict containing parameters will be inputted.
    Trades object to dict method can be used for documentation when a trade is closed
    """

    # trade parameters
    def __init__(self,
                 symbol: str,
                 entry_price: float,  # depends on current price
                 entry_date: pd.Timestamp,
                 quantity: float,  # can be negative
                 stop_loss_price: float,
                 take_profit_price: float,
                 tradeID: int,
                 commission: float, # % of commissions to be paid (both entry and exit/buy and sell)
                 commission_initial: float,
                 trade_type: str,
                 direction: str, #"LONG", "SHORT"
                 ):

        self.tradeID = tradeID

        # --- Core Trade Properties ---
        # all are defined when trade is called when signal
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.quantity = quantity
        self.status = "OPEN"  # Status can be 'OPEN' or 'CLOSED'
        self.slippage = 0 # NOT IN USE
        self.direction = direction #"LONG", "SHORT

        # Trade type (simple, trailing stop, ... etc)
        self.trade_type = trade_type  # will be expanded in the future Currently: "SIMPLE"

        # commission logic:
        # when trade is opened, will find out the total equity required to take the trade, called cost of trade
        # commission will simply be on top of that, in terms of pct, so (1 + commission_pct) * cost of trade
        self.commission = commission # commissions pct
        self.commission_initial = commission_initial  # absolute value of commission paid when trade is opened, added to total cost of transactions
        self.commission_final = 0 # absolute value of commission paid when trade is closed, deducted from profits

        # --- Risk Management ---
        # can be updated with update_sl/update_tp methods
        self.stop_loss = stop_loss_price # defined when given signal is generated and trade is executed
        self.take_profit = take_profit_price # defined when given signal is generated and trade is executed

        # --- PnL Tracking ---
        self.current_pnl = 0.0 # will be updated with update_pnl method

        # --- Exit Information (to be filled upon closing) ---
        self.exit_price = None
        self.exit_date = None
        self.exit_reason = None  # e.g., 'STOP LOSS', 'TAKE PROFIT', 'End of backtest'

    # print out trade details
    def __repr__(self):
        """
        A representation for easy printing of the Trade object.
        """
        return (
            f"Trade(ID={self.tradeID}, Status={self.status}, Entry={self.entry_price} @ {self.entry_date.strftime('%Y-%m-%d')}, "
            f"Qty={self.quantity}, SL={self.stop_loss}, TP={self.take_profit}, "
            f"PnL={self.current_pnl:.2f}, Exit={self.exit_price})")

    # close trade function
    def close_trade(self, exit_price: float, exit_date: pd.Timestamp, exit_reason: str, commission_final: float):
        """
        Method to close the trade.
        """
        if self.status == "OPEN":
            self.exit_price = exit_price
            self.exit_date = exit_date
            self.exit_reason = exit_reason
            # Basic PnL calculation (simplified for example)
            if self.direction == "LONG":
                self.current_pnl = (self.exit_price - self.entry_price) * self.quantity # commission calculation is in portfolio_management
            elif self.direction == "SHORT":
                self.current_pnl = (self.entry_price - self.exit_price) * self.quantity # commission calculation is in portfolio_management
            self.status = "CLOSED"
            self.commission_final = commission_final
        else:
            print(f"Trade {self.tradeID} is already closed.")

        # add COMMISSION_FINAL to trade when closed !!!!

    # update stop loss, take profit (NOT IN USE YET)
    def update_tp (self, updated_take_profit: float):
        self.take_profit = updated_take_profit

    def update_sl (self, update_stop_loss: float):
        self.stop_loss = update_stop_loss

    # updates pnl, current price will most likely use close price
    def update_pnl(self, current_price: float):
        self.current_pnl = (current_price - self.entry_price) * self.quantity - self.commission

    #create a dict type to display (closed) trade information
    def to_dict(self):
        """
        A dictionary representation for easy documenting of the Trade object. Used when a trade is
        closed, and such a dict is appended to a list
        """
        return {
            'symbol': self.symbol,
            'entry_price': round(self.entry_price, 2),
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'quantity': self.quantity,
            'direction' : self.direction,
            'stop_loss_price': round(self.stop_loss, 2),
            'take_profit_price': round(self.take_profit, 2),
            'PNL': round(self.current_pnl, 2),
            'tradeID': round(self.tradeID, 2),
            'commission': round(self.commission, 2),
            'commission_initial': round(self.commission_initial, 2),
            'commission_final': round(self.commission_final, 2),
            'exit_date': self.exit_date.strftime('%Y-%m-%d'),
            'exit_price': round(self.exit_price, 2),
            'status': "CLOSED",
            'exit_reason': self.exit_reason
        }

    # create a new trade from a dict, used when new trade is added
    @classmethod
    def create_from_dict(cls, trade_data: dict):
        """
        Creates a Trades object from a dictionary of trade arguments. Input must be a dict, predetermined
        by the function calling a new trades object

        Args:
            trade_data: A dictionary containing the arguments to create a Trades object.
            Must contain:
                symbol=trade_data['symbol'],
                entry_price=trade_data['entry_price'],
                entry_date=trade_data['entry_date'],
                quantity=trade_data['quantity'],
                stop_loss_price=trade_data['stop_loss_price'],
                take_profit_price=trade_data['take_profit_price'],
                tradeID=trade_data['tradeID'],
                commission=trade_data['commission'],
                commission_initial=trade_data['commission_initial'],
                trade_type=trade_data['trade_type'],
                direction=trade_data['direction'],
        Returns:
            A new Trades object.
        """
        # Unpack the dictionary into the Trades constructor using cls()
        return cls(
            symbol=trade_data['symbol'],
            entry_price=trade_data['entry_price'],
            entry_date=trade_data['entry_date'],
            quantity=trade_data['quantity'],
            stop_loss_price=trade_data['stop_loss_price'],
            take_profit_price=trade_data['take_profit_price'],
            tradeID=trade_data['tradeID'],
            commission=trade_data['commission'],
            commission_initial=trade_data['commission_initial'],
            trade_type=trade_data['trade_type'],
            direction=trade_data['direction'],
        )
