import sys

plugin = {
    "Name": "Face Segmentation (BiSeNet')",
    "Version": "0.1.0", 
    "Author": "DeepMake", 
    "Description": "Face segmentation using BiSeNet", 
    "env": "bisenet"
}
config = {
    "model_dtype": "fp32" if sys.platform == "darwin" else "fp16"
}
endpoints = {
    "segment_face": {
        "call": "execute",                             
        "inputs": {
            "img": "Image",
            "skin": "Bool(default=true, optional=true, help='whether to include skin in the mask')",
            "left_eybrow": "Bool(default=false, optional=true, help='whether to include left eyebrow in the mask')",
            "right_eyebrow": "Bool(default=false, optional=true, help='whether to include right eyebrow in the mask')",
            "left_eye": "Bool(default=false, optional=true, help='whether to include left eye in the mask')",
            "right_eye": "Bool(default=false, optional=true, help='whether to include right eye in the mask')",
            "eyeglasses": "Bool(default=false, optional=true, help='whether to include eye glasses in the mask')",
            "left_ear": "Bool(default=false, optional=true, help='whether to include left ear in the mask')",
            "right_ear": "Bool(default=false, optional=true, help='whether to include right ear in the mask')",
            "earring": "Bool(default=false, optional=true, help='whether to include ear ring(s) in the mask')",
            "nose": "Bool(default=false, optional=true, help='whether to include nose in the mask')",
            "mouth": "Bool(default=false, optional=true, help='whether to include the mouth mouth in the mask')",
            "upper_lip": "Bool(default=false, optional=true, help='whether to include upper lip in the mask')",
            "lower_lip": "Bool(default=false, optional=true, help='whether to include lower lip in the mask')",
            "neck": "Bool(default=false, optional=true, help='whether to include neck in the mask')",
            "necklace": "Bool(default=false, optional=true, help='whether to include necklace(s) in the mask')",
            "clothing": "Bool(default=false, optional=true, help='whether to include clothing in the mask')",
            "hair": "Bool(default=false, optional=true, help='whether to include hair in the mask')",
            "hat": "Bool(default=false, optional=true, help='whether to include hat(s) in the mask')"
        }, 
        "outputs": {"output_mask": "Image"}
    }
}