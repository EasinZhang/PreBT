import logging
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import torch
import numpy as np
import pandas as pd

def plot_results(config, predictions, targets, target_masks=None, normalizer=None):
    # Convert lists to NumPy arrays
    predictions_np = np.concatenate(predictions)
    targets_np = np.concatenate(targets)
    padding = np.concatenate(target_masks)

    # Convert NumPy arrays to PyTorch tensors
    predictions_tensor = torch.from_numpy(predictions_np)
    targets_tensor = torch.from_numpy(targets_np)
    padding_tensor = torch.from_numpy(padding)

    if config['data_class'] == 'socdataset':  # soc task
        # pred = torch.masked_select(predictions_tensor, padding_tensor).detach().cpu().numpy()
        # true = torch.masked_select(targets_tensor, padding_tensor).detach().cpu().numpy()
        pred = predictions_tensor
        true = targets_tensor

        plt.figure(figsize=(10, 5))
        plt.plot(pred, label='Predicted')
        plt.plot(true, label='True')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        #plt.show()
      # Compute MAE, RMSE, MAPE
        mae = torch.mean(torch.abs(pred - true))
        rmse = torch.sqrt(torch.mean((pred - true) ** 2))
        mape = torch.mean(torch.abs((pred - true) / torch.clamp(true, min=1e-9))) * 100

        print(f"MAE: {mae.item()}")
        print(f"RMSE: {rmse.item()}")
        print(f"MAPE: {mape.item()}")

        results_dict = {
            'MAE': [mae.item()],
            'RMSE': [rmse.item()],
            'MAPE': [mape.item()]
        }
        pred_np = pred.numpy()
        true_np = true.numpy()

        # 将NumPy数组转换为Pandas DataFrame
        df = pd.DataFrame({
            'Estimation': pred_np,
            'True': true_np
        })

        # 将DataFrame保存到CSV文件
        csv_file_path = r'C:\Users\Administrator\Desktop\results\SOE.csv'
        df.to_csv(csv_file_path, index=False)

        print(f"Data has been saved to {csv_file_path}")

        results_df = pd.DataFrame(results_dict)
        excel_file_path = r'E:\2_paper\mvts_transformer-master\SOC_results.xlsx'

        try:
            existing_data = pd.read_excel(excel_file_path)
            updated_data = pd.concat([existing_data, results_df], ignore_index=True)
        except FileNotFoundError:
            updated_data = results_df

        updated_data.to_excel(excel_file_path, index=False)

    elif config['data_class'] == 'sohdataset':  # soh task
        pred = predictions_tensor.detach().cpu()
        true = targets_tensor.detach().cpu()
        #存储结果

        # pred_np = pred.numpy()
        # true_np = true.numpy()
        #
        # # 将NumPy数组转换为Pandas DataFrame
        # df = pd.DataFrame({
        #     'Prediction': pred_np,
        #     'True': true_np
        # })
        #
        # # 将DataFrame保存到CSV文件
        # csv_file_path = r'C:\Users\Administrator\Desktop\results\MIT37_oneshot_SOH.csv'
        # df.to_csv(csv_file_path, index=False)
        #
        # print(f"Data has been saved to {csv_file_path}")



        pred = pred/3.2
        true = true/3.2
        # 创建一个包含两个子图的图形对象
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 第一个子图：时间步上的预测值与真实值的对比
        ax1.plot(pred, label='Predicted')
        ax1.plot(true, label='True')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.set_title('Predicted vs True over Time Steps')
        ax1.legend()

        # 第二个子图：真实值与预测值的关系图
        ax2.scatter(true, pred, label='Predictions')
        ax2.plot([min(true), max(true)], [min(true), max(true)], color='red', linestyle='--', label='Ideal Line')
        ax2.set_xlabel('True Value')
        ax2.set_ylabel('Predicted Value')
        ax2.set_title('True vs Predicted Values')
        ax2.legend()

        # 调整布局以避免重叠
        plt.tight_layout()

        # 显示图形
        plt.show()
        mae = torch.mean(torch.abs(pred - true))
        rmse = torch.sqrt(torch.mean((pred - true) ** 2))
        mape = torch.mean(torch.abs((pred - true) / torch.clamp(true, min=1e-9))) * 100

        print(f"MAE: {mae.item()}")
        print(f"RMSE: {rmse.item()}")
        print(f"MAPE: {mape.item()}")

        results_dict = {
            'MAE': [mae.item()],
            'RMSE': [rmse.item()],
            'MAPE': [mape.item()]
        }


        results_df = pd.DataFrame(results_dict)
        excel_file_path = r'E:\2_paper\mvts_transformer-master\SOH_results.xlsx'

        try:
            existing_data = pd.read_excel(excel_file_path)
            updated_data = pd.concat([existing_data, results_df], ignore_index=True)
        except FileNotFoundError:
            updated_data = results_df

        updated_data.to_excel(excel_file_path, index=False)

    elif config['data_class'] == 'sotdataset':  # soh task
        # pred = torch.masked_select(predictions_tensor, padding_tensor).detach().cpu().numpy()
        # true = torch.masked_select(targets_tensor, padding_tensor).detach().cpu().numpy()
        pred = predictions_tensor.flatten()
        true = targets_tensor.flatten()

        plt.figure(figsize=(10, 5))
        plt.plot(pred, label='Predicted')
        plt.plot(true, label='True')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        # Compute MAE, RMSE, MAPE
        mae = torch.mean(torch.abs(pred - true))
        rmse = torch.sqrt(torch.mean((pred - true) ** 2))
        mape = torch.mean(torch.abs((pred - true) / torch.clamp(true, min=1e-9))) * 100

        print(f"MAE: {mae.item()}")
        print(f"RMSE: {rmse.item()}")
        print(f"MAPE: {mape.item()}")

        results_dict = {
            'MAE': [mae.item()],
            'RMSE': [rmse.item()],
            'MAPE': [mape.item()]
        }

        results_df = pd.DataFrame(results_dict)
        excel_file_path = r'E:\2_paper\mvts_transformer-master\SOT_results.xlsx'

        try:
            existing_data = pd.read_excel(excel_file_path)
            updated_data = pd.concat([existing_data, results_df], ignore_index=True)
        except FileNotFoundError:
            updated_data = results_df

        updated_data.to_excel(excel_file_path, index=False)

    else:
        mask_feat = len(config['mask_feats'])
        # Flatten the predictions for normalization
        predictions_np_flat = predictions_np.reshape(-1, predictions_np.shape[-1])
        targets_np_flat = targets_np.reshape(-1, targets_np.shape[-1])

        pred_df = pd.DataFrame(predictions_np_flat)
        true_df = pd.DataFrame(targets_np_flat)

        pred = normalizer.denormalize(pred_df).values
        true = normalizer.denormalize(true_df).values

        pred = pred.reshape(-1, predictions_np.shape[1], predictions_np.shape[2])
        true = true.reshape(-1, targets_np.shape[1], targets_np.shape[2])

        pred_tensor = torch.from_numpy(pred)
        true_tensor = torch.from_numpy(true)

        for i in range(4):
            pred_masked = torch.masked_select(pred_tensor[:, :, i], padding_tensor[:, :, i])
            true_masked = torch.masked_select(true_tensor[:, :, i], padding_tensor[:, :, i])
            # plt.figure(figsize=(10, 5))
            # plt.plot(pred_masked[:2000], label='Predicted')
            # plt.plot(true_masked[:2000], label='True')
            # plt.xlabel('Time Step')
            # plt.ylabel('Value')
            # plt.legend()
            # plt.show()

            # Compute MAE, RMSE, MAPE
            mae = torch.mean(torch.abs(pred_masked - true_masked))
            rmse = torch.sqrt(torch.mean((pred_masked - true_masked) ** 2))
            mape = torch.mean(torch.abs((pred_masked - true_masked) / torch.clamp(true_masked, min=1e-9))) * 100

            print(f"MAE: {mae.item()}")
            print(f"RMSE: {rmse.item()}")
            print(f"MAPE: {mape.item()}")

            results_dict = {
                'MAE': [mae.item()],
                'RMSE': [rmse.item()],
                'MAPE': [mape.item()]
            }

            if i == 0:
                results_df = pd.DataFrame(results_dict)
                excel_file_path = r'E:\2_paper\mvts_transformer-master\results.xlsx'

                try:
                    existing_data = pd.read_excel(excel_file_path)
                    updated_data = pd.concat([existing_data, results_df], ignore_index=True)
                except FileNotFoundError:
                    updated_data = results_df

                updated_data.to_excel(excel_file_path, index=False)

            # Different colors for i == 0 and i == 4
            if i == 0:
                pred_color = '#287271'  # Softer blue for predicted
                true_color = '#F4931E'  # Softer orange for true
            elif i == 3:
                pred_color = '#287271'  # Green for predicted
                true_color = '#F4931E'  # Reddish-orange for true

            if i in [0, 3]:  # Save plots at i == 0 and i == 4
                plt.figure(figsize=(10, 6))
                plt.plot(pred_masked[:2000], label='Estimation', markersize=3, color= pred_color)
                plt.plot(true_masked[:2000], label='True',  linestyle='--', markersize=3, color=true_color)

                if i==0:
                   plt.ylim(2.4, 4.6)
                   plt.xlabel('Sampling Point', fontsize=12)
                   plt.ylabel('Voltage (V)', fontsize=12)
                   plt.title('Estimation and True Voltage', fontweight='bold',fontsize=16)
                elif i==3:
                    plt.xlabel('Sampling Point', fontsize=12)
                    plt.ylabel('Current (A)', fontsize=12)
                    plt.title('Estimation and True Current', fontweight='bold',fontsize=16)
                plt.legend(fontsize=14, loc='upper right')

                # Save tiff image
                tiff_file_path = r'C:\Users\Administrator\Desktop\results\自监督\plot_results_i{}.tiff'.format(i)
                plt.savefig(tiff_file_path, format='tiff', bbox_inches='tight', dpi=300)
                plt.show()