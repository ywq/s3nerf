from .base_opts import BaseOpts
class RunModelOpts(BaseOpts):
    def __init__(self):
        super(RunModelOpts, self).__init__()
        self.initialize()

    def initialize(self):
        BaseOpts.initialize(self)
        #### Testing Dataset Arguments #### 
        self.parser.add_argument('--run_model',  default=True, action='store_false')
        self.parser.add_argument('--benchmark',  default='UPS_DiLiGenT_main')
        self.parser.add_argument('--bm_dir',     default='data/datasets/DiLiGenT/pmsData_crop')
        self.parser.add_argument('--epochs',     default=1,   type=int)
        self.parser.add_argument('--test_batch', default=1,   type=int)
        self.parser.add_argument('--test_disp',  default=1,   type=int)
        self.parser.add_argument('--test_save',  default=1,   type=int)
        
        self.parser.add_argument('--light_intnorm_gt',   default=False, action='store_true')
        self.parser.add_argument('--train_light',  default=None,   type=int)

    def collectInfo(self):
        self.args.str_keys  = ['model', 'model_s2', 'benchmark', 'fuse_type']
        self.args.val_keys  = ['in_img_num', 'test_h', 'test_w']
        self.args.bool_keys = ['int_aug', 'test_resc']
    def setDefault(self):
        self.collectInfo()

    def parse(self):
        BaseOpts.parse(self)
        self.setDefault()
        return self.args
