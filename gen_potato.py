import io, sys, os, pickle, base64, torch, torchvision
from generator import Generator

num_gpu = 0 # number of GPUs available. Use 0 for CPU mode.

# define device
device = torch.device("cpu")
Generator(num_gpu=0).to(device)

# collect model files
model_dir = 'model'
g_path = os.path.join(model_dir, 'G.pkl')

class DeviceUnpickler(pickle.Unpickler):
    '''unpickle on current device (CPU or GPU), independent of which device the model was trained on'''
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else: return super().find_class(module, name)

def gen_tater(model):
    '''generate 4 random fake images with given model then select the 0th item'''
    noise = torch.randn(4, 100, 1, 1, device=device)
    fakes = model(noise).detach().cpu()
    fake_img = fakes[0]
    grid = torchvision.utils.make_grid(fake_img, padding=2, normalize=True)
    img = torchvision.transforms.ToPILImage()(grid)
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return img_str

def make_tater():
    '''load model files'''

    g_file = open(g_path, 'rb')
    netG = DeviceUnpickler(g_file).load()
    g_file.close()

    # send model to CPU
    netG = netG.module.to(device)

    return gen_tater(netG)

tater = make_tater()
sys.stdout.write(tater)