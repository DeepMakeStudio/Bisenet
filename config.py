import sys

plugin = {
    "Name": "Face Segmentation (BiSeNet)",
    "Version": "0.1.0", 
    "Author": "DeepMake", 
    "Description": "Face segmentation using BiSeNet", 
    "env": "bisenet"
}
config = {
    "model_name": "plugin/bisenet/res/cp/79999_iter.pth",
    "model_dtype": "fp32" if sys.platform == "darwin" else "fp16"
}
endpoints = {
    "segment_face": {
        "call": "execute",                             
        "inputs": {
            "img": "Image",
            "skin": "Bool(default=true, optional=true, help='whether to include skin in mask)",
            "l_brow": "Bool(default=false, optional=true, help='whether to include left eyebrow in mask)",
            "r_brow": "Bool(default=false, optional=true, help='whether to include right eyebrow in mask)",
            "l_eye": "Bool(default=false, optional=true, help='whether to include left eye in mask)",
            "r_eye": "Bool(default=false, optional=true, help='whether to include right eye in mask)",
            "eye_g": "Bool(default=false, optional=true, help='whether to include eye glasses in mask)",
            "l_ear": "Bool(default=false, optional=true, help='whether to include left ear in mask)",
            "r_ear": "Bool(default=false, optional=true, help='whether to include right ear in mask)",
            "ear_r": "Bool(default=false, optional=true, help='whether to include ear ring in mask)",
            "nose": "Bool(default=false, optional=true, help='whether to include nose in mask)",
            "mouth": "Bool(default=false, optional=true, help='whether to include mouth in mask)",
            "u_lip": "Bool(default=false, optional=true, help='whether to include upper lip in mask)",
            "l_lip": "Bool(default=false, optional=true, help='whether to include lower lip in mask)",
            "neck": "Bool(default=false, optional=true, help='whether to include neck in mask)",
            "neck_l": "Bool(default=false, optional=true, help='whether to include necklaces in mask)",
            "cloth": "Bool(default=false, optional=true, help='whether to include clothing in mask)",
            "hair": "Bool(default=false, optional=true, help='whether to include hair in mask)",
            "hat": "Bool(default=false, optional=true, help='whether to include hats in mask)"
        }, 
        "outputs": {"output_img": "Image"}
    }
}

['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']