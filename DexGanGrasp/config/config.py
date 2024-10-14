import datetime
import os
import shutil

import yaml

# If you want to continue training, set continue_train=True, start_epoch=desired_epoch and load_path=/path/to/pretrained
path = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(path)[0])[0]

class Config(object):
    def __init__(self,config_path):
        self.abs_config_path = os.path.join(BASE_PATH, config_path)
        with open(self.abs_config_path, "r") as stream:
            try:
                self.cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def parse(self):
        if self.cfg["is_train"]:
            # If you want to write your own file name
            # name = 'ffhnet' + '_lr_' + str(self.cfg["lr"]) + '_bs_' + str(
            #     self.cfg["batch_size"]) + '_scale_' + str(self.cfg["scale"]) + '_latentd_' + str(self.cfg["latentD"])
            # self.cfg["name"] = name

            # create and set checkpoints dir if training from scratch. Otherwise set load directory as ckpts directory.
            if self.cfg["continue_train"]:
                self.cfg["save_dir"] = self.cfg["load_path"]
            else:
                ckpts_dir = os.path.join(BASE_PATH, 'checkpoints')
                if not os.path.exists(ckpts_dir):
                    os.mkdir(ckpts_dir)

                # Create folder with datetime.time as name und ckpts dir
                now = datetime.datetime.now().replace(microsecond=0).isoformat().replace(':', '_')
                folder_name = now + '_' + self.cfg["name"]
                self.cfg["save_dir"] = os.path.join(ckpts_dir, folder_name)
                os.mkdir(self.cfg["save_dir"])

                # Save the config
                yaml_path = os.path.join(self.cfg["save_dir"], 'config.yaml')
                with open(yaml_path, 'w') as yaml_file:
                    yaml.dump(self.cfg, yaml_file)
        else: # eval mode
            self.cfg["save_dir"] = self.cfg["load_path"]

        # copy the config file to save_dir
        fname = os.path.join(self.cfg["save_dir"],'config_default.yaml')
        if not os.path.isfile(fname):
            shutil.copy(self.abs_config_path,fname)

        # Create eval dir
        self.cfg["eval_dir"] = os.path.join(self.cfg["save_dir"], 'eval')
        if not os.path.exists(self.cfg["eval_dir"]):
            os.mkdir(self.cfg["eval_dir"])

        return self.cfg
