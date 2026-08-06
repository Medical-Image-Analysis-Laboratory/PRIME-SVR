# import debugpy

# Listen for connections from Cursor (make sure the port matches your launch.json)
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")

# # Wait for Cursor to connect before proceeding
# debugpy.wait_for_client()
# print("Debugger attached!")

import argparse
import os
import yaml
from sv_reconstruction import SVR_Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_new.yaml')
    parser.add_argument('--path_stacks', type=str)
    parser.add_argument('--path_masks', type=str)
    parser.add_argument('--batch_size', type=int)
    # parameters needed: epochs, psf_card_end, sr_omega_0, sr_n_hiidden_layer, sr_n_uits, sr_loss_metrics, outlier rejection, slice_weight_reg, inr
    parser.add_argument('--path_output', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--psf_card_end', type=int)
    parser.add_argument('--sr_omega_0', type=float)
    parser.add_argument('--sr_n_hidden_layers', type=int)
    parser.add_argument('--sr_n_units', type=int)
    parser.add_argument('--sr_loss_metric', type=str)
    parser.add_argument('--outlier_rejection', type=str)
    parser.add_argument('--slice_weight_reg', type=float)
    parser.add_argument('--inr', type=str) #siren or hashgrid for SR module
    return parser.parse_args()

def get_config():
    cwd = os.getcwd().replace('/SirenSVR', '')
    args = parse_args()

    #with open(cwd + '/SirenSVR/configs/' + args.config, 'r') as f:
    #    config = yaml.safe_load(f)
    #config_path = "/home/mroulet/Documents/Data/fqMRI/low_field/chuv001/output_SIREN/configs/MC_reg_4/config_MC_reg_4.yaml"
    #with open(config_path, 'r') as f:
        #config = yaml.safe_load(f)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # merge args and config into one dict where args overwrite config if provided
    merged_config = {}
    for key, value in config.items():
        # args is a namespace object, so we need to access the attributes via getattr
        if key in args and getattr(args, key) is not None:
            merged_config[key] = getattr(args, key)
        else:
            merged_config[key] = value
    return merged_config

if __name__ == "__main__":
    print("Running PRIME-SVR")
    merged_config = get_config()
    os.makedirs(merged_config['data']['recon_dir'], exist_ok=True)
    svr_trainer = SVR_Trainer(merged_config)
    svr_trainer.train()
    if merged_config.get('data', {}).get('simulated_slices', False):
        svr_trainer.reconstruct_simulated_stacks(save_prefix="simulated_stack")
    #save the model 
    model_path = os.path.join(merged_config['data']['recon_dir'],"svr_model.pth")
    svr_trainer.save_model(model_path)



#tmux new-session -d -s run_SVR ' python run_new.py --config /home/mroulet/Documents/Data/KCL_clean/cat005/output_SIREN/configs/MC_reg/config_MC_reg_red_2_smart.yaml> run_SVR.log 2>&1'