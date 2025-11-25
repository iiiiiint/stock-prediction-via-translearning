from src.config import get_default_config
from src.data_manager import DataManager
cfg = get_default_config()
dm = DataManager(cfg)
dm.load_all_raw_data()
dm.align_merge_target()
dm.generate_target_features_base()
df = dm.get_target_df()
print('data rows:', len(df))
print('start_index_for_backtest:', cfg['start_index_for_backtest'])
print('valid_len:', cfg['valid_len'])
print('seq_len:', cfg['seq_len'])
print('dates head:', list(df.index[:5]))
print('dates tail:', list(df.index[-5:]))