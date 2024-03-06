
import torch
def convert_weights(source_model, output_model):
    source_weights = torch.load(source_model, map_location=torch.device('cpu'))
    converted_weights = {}
    keys = list(source_weights.keys())
    print(keys)

    prefix = 'model.byteformer.'
    for key in keys:
        if not key.startswith('classifier.'):
            print(key)
            newkey = prefix + key

            # print(newkey)
            converted_weights[newkey] = source_weights[key]

    torch.save(converted_weights, output_model)

def convert_weights_deit(source_model, output_model):
    source_weights = torch.load(source_model, map_location=torch.device('cpu'))
    converted_weights = {}
    converted_model ={"model" : converted_weights}
    source_weights = source_weights['model']
    keys = list(source_weights.keys())
    print(keys)

    #prefix = 'model.byteformer.'
    for key in keys:
        if  key.startswith('blocks.'):
            print(key)
            newkey = 'byteformer.transformer.' + key

            converted_weights[newkey] = source_weights[key]
        elif  key=='norm.weight':
            print(key)
            newkey = 'byteformer.post_transformer_norm.weight'

            converted_weights[newkey] = source_weights[key]
        elif  key=='norm.bias':
            print(key)
            newkey = 'byteformer.post_transformer_norm.bias'


            converted_weights[newkey] = source_weights[key]
 
    torch.save(converted_model, output_model)

def print_weights(source_model):
    source_weights = torch.load(source_model, map_location=torch.device('cpu'))
    converted_weights = {}
    source_weights = source_weights['model']
    keys = list(source_weights.keys())
    print(keys)

    

if __name__ == "__main__":
    #convert_weights('/data/train_logs/byteformer_imagen_h264_v20230930_wfan_2023-11-05-19-15-15/best_model.pt')
    #convert_weights('/data/speech_commands_wav.pt', '/data/speech_commands_wav_converted.pt')
    #convert_weights_deit('/data/deit_tiny_patch16_224-a1311bcf.pth', '/data/deit_tiny_patch16_224-a1311bcf.pth_convertednew.pt')
    #convert_weights_deit('/data/deit_small_patch16_224-cd65a155.pth', '/data/deit_small_patch16_224-cd65a155_convertednew.pth')
    #convert_weights_deit('/data/deit_base_patch16_224-b5f2ef4d.pth', '/data/deit_base_patch16_224-b5f2ef4d_convertednew.pth')
    #print_weights('/data/train_logs/byteformer_imagen_h264_v20230930-v20231107_wfan_2023-11-09-23-32-22/checkpoint.9999.last.pt')

   