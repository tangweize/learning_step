import pandas as pd

def add_date_features(df, promo_dates, holiday_dates):
    df['dt'] = pd.to_datetime(df['dt'])
    df['month'] = df['dt'].dt.month
    df['weekday'] = df['dt'].dt.weekday

    promo_set = set(pd.to_datetime(promo_dates).date)
    holiday_set = set(pd.to_datetime(holiday_dates).date)

    df['is_holiday'] = df['dt'].dt.date.apply(lambda x: 1 if x in holiday_set else 0)
    df['is_promo'] = df['dt'].dt.date.apply(lambda x: 1 if x in promo_set else 0)
    df['days_to_next_promo'] = df['dt'].apply(lambda x: days_to_next_promo(x, promo_set, holiday_set))
    df['days_past_last_promo'] = df['dt'].apply(lambda x: days_past_last_promo(x, promo_set, holiday_set))
    df['season'] = df['month'].apply(get_season)
    return df

def days_to_next_promo(current_date, promo_set, holiday_set):
    future_dates = [d for d in promo_set.union(holiday_set) if d >= current_date.date()]
    if not future_dates:
        return 0
    delta = min((d - current_date.date()).days for d in future_dates)
    return delta if delta <= 10 else 0

def days_past_last_promo(current_date, promo_set, holiday_set):
    past_dates = [d for d in promo_set.union(holiday_set) if d < current_date.date()]
    if not past_dates:
        return 0
    delta = min((current_date.date() - d).days for d in past_dates)
    return delta if delta <= 10 else 0

def get_season(month):
    return ('spring' if month in [3,4,5] else
            'summer' if month in [6,7,8] else
            'autumn' if month in [9,10,11] else 'winter')


def process_label(df, label_col, diff_col, target):
    df[target] = df[label_col] - df[diff_col]
    return df
