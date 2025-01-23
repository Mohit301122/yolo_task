model_preprocessing = {"lenet"       : {"image_size": (32, 32),  # LeNet-5 was designed for 32x32 images
                                        "normalize" : {"mean" : [0.5],
                                                       "std"  : [0.5]
                                                      } 
                                       },
                       "alexnet"     : {"image_size" : (224, 224),  # AlexNet expects 224x224 images
                                        "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
                                                        "std"  : [0.229, 0.224, 0.225]
                                                       }  
                                       },
                       "zfnet"       : {"image_size" : (224, 224),  # ZFNet also expects 224x224 images (similar to AlexNet)
                                        "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
                                                        "std"  : [0.229, 0.224, 0.225]
                                                       }  
                                       },
                       "vgg"         : {"image_size" : (224, 224),  # VGG models also use 224x224 images
                                        "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
                                                        "std"  : [0.229, 0.224, 0.225]
                                                       } 
                                       },
                       "efficientnet": {"image_size" : (224, 224),  # EfficientNet typically uses 224x224 input size
                                        "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
                                                        "std"  : [0.229, 0.224, 0.225]
                                                       }  
                                       },
                       "googlenet"   : {"image_size" : (224, 224),  # GoogleNet uses 224x224 input size
                                        "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
                                                        "std"  : [0.229, 0.224, 0.225]
                                                       } 
                                       },
                       "resnet"      : {"image_size" : (224, 224),  # ResNet models typically use 224x224 input size
                                        "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
                                                        "std"  : [0.229, 0.224, 0.225]
                                                       }  
                                       },
                       "shufflenet"  : {"image_size" : (224, 224),  # ShuffleNet uses 224x224 images
                                        "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
                                                        "std"  : [0.229, 0.224, 0.225]
                                                       }  
                                       },
                       "squeezenet"  : {"image_size" : (224, 224),  # SqueezeNet typically uses 224x224 input size
                                        "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
                                                        "std"  : [0.229, 0.224, 0.225]
                                                       } 
                                       },
                       "mobilenet"   : {"image_size" : (224, 224),  # MobileNet models use 224x224 images
                                        "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
                                                        "std"  : [0.229, 0.224, 0.225]
                                                       }
                                       },
                       "unet"        : {"image_size" : (224, 224),  # unet models use 256x256 images
                                        "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
                                                        "std"  : [0.229, 0.224, 0.225]
                                                       }
                                       }
                      }
