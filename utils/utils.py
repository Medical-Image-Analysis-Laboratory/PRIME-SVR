
import os 
def get_subject_name(path):
    if 'dhcp_fetal' in path:
        # sample path: /data/projects/dhcp_fetal/raw_stacks/total/test/experiment2/CC00862XX13/ses-41210/anat/sub-CC00862XX13_ses-41210_run-07_T2w.nii.gz
        # find "/ses-" and extract the subject id
        sub_id = "sub-" + path.split('/ses-')[0].split('/')[-1].replace('sub-', '')
        ses_id = path.split('/ses-')[1].split('/')[0]
    elif 'multi_modal_simulated' in path:
        sub_id = "data_steven"
        ses_id = "multi_modal_simulated"
    elif 'multi_modal_busra' in path:
        # path = "/vol/miltank/users/danneckm/Datasets/multi_modal_busra/anat-3/sub-002_ses-01_run-01_T2w.nii.gz"
        sub_id = path.split('/')[-1].split('_')[0]
        ses_id = path.split('/')[-1].split('_')[1]
    elif 'MarsFet' in path:
        sub_id = path.split('/')[-1].split('_')[0]
        ses_id = path.split('/')[-1].split('_')[1]
    # add additional rules here /data/projects/data_fetal_TE_KCL/cat002_TE1_ax.nii.gz
    elif 'fqMRI' in path:
        #sub_id = path.split('/')[-1].split('_')[0]
        #ses_id = path.split('/')[-1].split('_')[1]
        sub_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        #ses_id=""
        return sub_id 
    else:
        # path = "/vol/miltank/users/danneckm/Datasets/multi_modal_busra/anat-3/sub-002_ses-01_run-01_T2w.nii.gz"
        sub_id = path.split('/')[-1].split('_')[0]
        ses_id = path.split('/')[-1].split('_')[1]
        #sub_id = path.split('/')[-1].split('.')[0]
        #ses_id=""
        return sub_id + "_" + ses_id
    return sub_id + "_" + ses_id
