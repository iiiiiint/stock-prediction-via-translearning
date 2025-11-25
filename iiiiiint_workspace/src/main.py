from __future__ import annotations

from src.config import get_default_config
from src.data_manager import DataManager
from src.train_lstm import train_lstm_single
from src.train_lightgbm import train_lightgbm_single
from src.train_ndx import generate_ndx_predictions
from src.transfer_learning import train_transfer_learning, prepare_transfer_data
from src.ensemble_model import EnsembleModel, evaluate_ensemble_performance
from src.services import RollingLLMService

def main():
    config = get_default_config()
    
    print("请选择训练模式:")
    print("1. 单独模型训练")
    print("2. 集成模型训练")
    print("3. 完整流程比较")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 单独模型训练
        print("\n选择要训练的模型:")
        print("1. LSTM模型")
        print("2. LightGBM模型")
        print("3. 迁移学习模型")
        print("4. 美股预测模型")
        print("5. LLM专家模型")
        
        model_choice = input("请输入选择 (1/2/3/4/5): ").strip()
        
        if model_choice == "1":
            dm = DataManager(config)
            dm.load_all_raw_data()
            dm.align_merge_target()
            results = train_lstm_single(config, dm)
            
        elif model_choice == "2":
            dm = DataManager(config)
            dm.load_all_raw_data()
            dm.align_merge_target()
            results = train_lightgbm_single(config, dm)
            
        elif model_choice == "3":
            dm_target = DataManager(config)
            dm_target.load_all_raw_data()
            dm_target.align_merge_target()
            
            source_config = config.copy()
            source_config["data_sources"]["target"]["filename"] = "sourcedata.xlsx"
            dm_source = DataManager(source_config)
            dm_source.load_all_raw_data()
            dm_source.align_merge_target()
            
            results = train_transfer_learning(config, dm_target, dm_source)
            
        elif model_choice == "4":
            source_config = config.copy()
            source_config["data_sources"]["target"]["filename"] = "sourcedata.xlsx"
            dm_source = DataManager(source_config)
            dm_source.load_all_raw_data()
            dm_source.align_merge_target()
            
            dm_target = DataManager(config)
            results = generate_ndx_predictions(config, dm_source, dm_target)
            
        elif model_choice == "5":
            dm = DataManager(config)
            dm.load_all_raw_data()
            dm.align_merge_target()
            llm_service = RollingLLMService(config, dm)
            # LLM不需要训练，直接验证
            
        else:
            print("无效选择")
            return
        
        if 'results' in locals():
            print(f"\n训练完成，结果行数: {len(results)}")
            results.to_csv(f"results_single_model.csv")
            print("结果已保存到 results_single_model.csv")
            
    elif choice == "2":
        # 集成模型训练
        print("\n开始集成模型训练...")
        
        ensemble = EnsembleModel(config)
        
        # 训练各个模型
        dm_target = DataManager(config)
        dm_target.load_all_raw_data()
        dm_target.align_merge_target()
        
        # LSTM
        lstm_results = train_lstm_single(config, dm_target)
        ensemble.add_model_results("LSTM", lstm_results)
        
        # LightGBM
        dm_target.generate_advanced_lgbm_features()
        lgbm_results = train_lightgbm_single(config, dm_target)
        ensemble.add_model_results("LightGBM", lgbm_results)
        
        # 迁移学习
        source_config = config.copy()
        source_config["data_sources"]["target"]["filename"] = "sourcedata.xlsx"
        dm_source = DataManager(source_config)
        dm_source.load_all_raw_data()
        dm_source.align_merge_target()
        
        transfer_results = train_transfer_learning(config, dm_target, dm_source)
        ensemble.add_model_results("Transfer", transfer_results)
        
        # 美股预测
        ndx_results = generate_ndx_predictions(config, dm_source, dm_target)
        ensemble.add_model_results("NDX", ndx_results)
        
        # 运行集成
        final_results = ensemble.run_ensemble()
        metrics = evaluate_ensemble_performance(final_results)
        
        print("\n===== 集成模型表现 =====")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        final_results.to_csv("results_ensemble.csv")
        print("集成结果已保存到 results_ensemble.csv")
        
    elif choice == "3":
        # 完整流程比较（原有逻辑）
        from src.experiments import main_pipeline_with_comparison
        main_pipeline_with_comparison(config)
    
    else:
        print("无效选择")

if __name__ == "__main__":
    main()