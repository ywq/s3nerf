from .base_opts import BaseOpts
class TrainOpts(BaseOpts):
    def __init__(self):
        super(TrainOpts, self).__init__()
        self.initialize()

    def initialize(self):
        BaseOpts.initialize(self)
        #### Training Arguments ####
        self.parser.add_argument('--solver',      default='adam', help='adam|sgd')
        self.parser.add_argument('--milestones',  default=[5, 10, 15, 20, 25], nargs='+', type=int)
        self.parser.add_argument('--start_epoch', default=1,      type=int)
        self.parser.add_argument('--epochs',      default=20,     type=int)
        self.parser.add_argument('--batch',       default=32,     type=int)
        self.parser.add_argument('--val_batch',   default=8,      type=int)
        self.parser.add_argument('--init_lr',     default=0.0005, type=float)
        self.parser.add_argument('--lr_decay',    default=0.5,    type=float)
        self.parser.add_argument('--beta_1',      default=0.9,    type=float, help='adam')
        self.parser.add_argument('--beta_2',      default=0.999,  type=float, help='adam')
        self.parser.add_argument('--momentum',    default=0.9,    type=float, help='sgd')
        self.parser.add_argument('--w_decay',     default=4e-4,   type=float)

        #### Loss Arguments ####
        self.parser.add_argument('--normal_loss', default='cos',  help='cos|mse')
        self.parser.add_argument('--normal_w',    default=1,      type=float)
        self.parser.add_argument('--dir_loss',    default='cos',  help='cos|mse')
        self.parser.add_argument('--dir_w',       default=1,      type=float)
        self.parser.add_argument('--ints_loss',   default='mse',  help='l1|mse')
        self.parser.add_argument('--ints_w',      default=1,      type=float)

    def collectInfo(self): 
        BaseOpts.collectInfo(self)
        self.args.str_keys  += [
                'dir_loss',
                ]
        self.args.val_keys  += [
                ]
        self.args.bool_keys += [
                ] 

    def setDefault(self):
        BaseOpts.setDefault(self)
        if self.args.test_h != self.args.crop_h:
            self.args.test_h, self.args.test_w = self.args.crop_h, self.args.crop_w
        self.collectInfo()

    def parse(self):
        BaseOpts.parse(self)
        self.setDefault()
        return self.args
