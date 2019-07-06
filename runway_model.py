import runway
from runway.data_types import number, text, image
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F


# the architecture to use
arch = 'resnet18'
# load the class label
file_name = 'categories_places365.txt'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@runway.setup
def setup():
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

@runway.command('classify', inputs={'photo': image}, outputs={'label': text})
def classify(model, inputs):
    img = inputs['photo']
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    # output the prediction
    for i in range(0, 5):
        the_text = str(classes[idx[i]]).replace("_", " ")
        print('{:.3f} -> {}'.format(probs[i], the_text))

    output_prob = '{:.3f}'.format(probs[0])
    output_label = '{}'.format(str(classes[idx[0]]))

    return {'label': output_label}

    
if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run(host='0.0.0.0', port=8000)

## Now that the model is running, open a new terminal and give it a command to
## generate an image. It will respond with a base64 encoded URI
# curl \
#   -H "content-type: application/json" \
#   -d '{ "image": "red" }' \
#   localhost:8000/generate