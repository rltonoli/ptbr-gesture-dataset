# FROM: https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/blender_rendering/utils/fbx2bvh.py

import bpy
import numpy as np
from os import listdir, path, getcwd

def fbx2bvh(data_path, file):
    sourcepath = data_path+"/"+file
    bvh_path = data_path+"/"+file.split(".fbx")[0]+".bvh"

    bpy.ops.import_scene.fbx(filepath=sourcepath)

    frame_starts = []
    frame_ends = []
    for a in bpy.data.actions:
        frame_starts.append(a.frame_range[0])
        frame_ends.append(a.frame_range[1])
    frame_start = np.min(frame_starts)
    frame_end = np.max(frame_ends)
      
    print(frame_start, frame_end)

    #frame_end = np.max([60, frame_end])
    bpy.ops.export_anim.bvh(filepath=bvh_path,
                            frame_start= int(frame_start),
                            frame_end=int(frame_end), root_transform_only=True)
    bpy.data.actions.remove(bpy.data.actions[-1])
    print(data_path+"/"+file+" processed.")

if __name__ == '__main__':
    data_path = "C:\\Users\\Rodolfo\\Desktop\\teste\\"
    print("cwd:")
    print(getcwd())
    files = sorted([f for f in listdir(data_path) if f.endswith(".fbx")])
    print("files:")
    print(files)
    for file in files:
      fbx2bvh(data_path, file)
      #clear scene
      for c in bpy.context.scene.collection.children:
          for o in c.objects:
              bpy.data.objects.remove(o)
      for a in bpy.data.actions:
          bpy.data.actions.remove(a)