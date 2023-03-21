from torch.utils.data import Dataset
from os import listdir, path
from pickle import load
import numpy as np

class GestData(Dataset):
    def __init__(self, 
                npy_motion_path,
                npy_audio_path,
                window=120,
                overlap=60,
                ):

        self.window = window
        self.overlap = overlap
        self.classes = 0

        motionfiles = listdir(npy_motion_path)
        audiofiles = listdir(npy_audio_path)
        motionfiles = [file for file in motionfiles if file[-4:]==".npy"]
        motionfiles.sort()
        audiofiles.sort()

        # Check the number of wav and npy files
        assert len(motionfiles) == len(audiofiles)
        self.takes = []

        # Creates a take for each pair of file
        for motionfile, audiofile in zip(motionfiles, audiofiles):
            assert motionfile.split('.')[0] == audiofile.split('.')[0]
            samp = Take(motionfile, audiofile, npy_motion_path, npy_audio_path)
            self.takes.append(samp)

        # Get joint names
        with open(path.join(npy_motion_path, 'joints'), 'rb') as f:
            self.jointnames = load(f)
        self.joint = {k+s:v*6+j for v,k in enumerate(self.jointnames) for j,s in enumerate(["_rx", "_ry", "_rz", "_tx", "_ty", "_tz"])}


        #self.createLabels()
        #self.maxs, self.mins = self.get_maxmin()
        #self.feature_scaling()
        #self.prepare_samples2()

    def __getitem__(self, index):
        return self.input[index], self.label[index]

    def __len__(self):
        return self.input.shape[0]

    def downsample(self, target_fps, current_fps):
        assert current_fps%target_fps==0
        ratio = int(current_fps/target_fps)
        for take in self.takes:
            take.motion.data = take.motion.data[::ratio,:]
            #TODO: FAZER PARA ÁUDIO

    def index2take(self, index):
        found = False
        take = 0
        while not found:
            if self.samples_per_take[take] > index:
                found = True
            else:
                index -= self.samples_per_take[take]
                take += 1
        return take, index #return take and index inside the take

    def prepare_samples(self):
        # Get how many frames each take has
        self.frameslist = [take.motion.data.shape[0] for take in self.takes]
        # Get index of each sample for each take
        self.samples_index_per_take = []
        for frames_in_take in self.frameslist:
            self.samples_index_per_take.append( [(i-self.window, i) for i in np.arange(self.window, frames_in_take, self.window-self.overlap)] )
        # Samples per take
        self.samples_per_take = [len(i) for i in self.samples_index_per_take]
        # Get samples per take cumulative
        self.samples_per_take_acc = [ np.sum(self.samples_per_take[:i+1]) for i in np.arange(len(self.samples_per_take))]
        total_samples = self.samples_per_take_acc[-1]

        self.input = np.zeros(shape=(total_samples, self.window, 360))
        self.label = np.zeros(shape=(total_samples, self.classes))
        for i in range(total_samples):
            take, index = self.index2take(i)
            sample_range = self.samples_index_per_take[take][index]
            self.input[ i, :, :] = self.takes[take].motion.data[sample_range[0]:sample_range[1],:]
            self.label[i] = self.takes[take].takelabel
        #self.input = torch.as_tensor(self.input, dtype=torch.float)
        #self.label = torch.as_tensor(self.label, dtype=torch.float)


    def feature_scaling(self):
        for take in self.takes:
            take.motion.data = np.nan_to_num((take.motion.data - self.mins) / (self.maxs - self.mins), nan=0.0, copy=False)


    def compute_maxmin(self):
        maxs = np.zeros(360)-np.inf
        mins = np.zeros(360)+np.inf
        for take in self.takes:
            aux_max = np.max(take.motion.data, axis=0)
            aux_min = np.min(take.motion.data, axis=0)
            maxs = np.maximum(maxs, aux_max)
            mins = np.minimum(mins, aux_min)
        self.maxs = maxs
        self.mins = mins
        return maxs, mins

    def compute_relative_positions(self):
        # Compute the position of each joint relative to the hips
        positions_indexes = [i+j*6 for j,_ in enumerate(range(60)) for i in [3,4,5]]
        hips_centric_indexes = positions_indexes[:36] + positions_indexes[90:102] + positions_indexes[156:]
        rhand_centric_indexes = positions_indexes[36:90]
        lhand_centric_indexes = positions_indexes[102:156]

        for take in self.takes:
            for i in range(3,len(hips_centric_indexes), 3):
                take.motion.data[:,hips_centric_indexes[i:i+3]] = take.motion.data[:,hips_centric_indexes[i:i+3]] - take.motion.data[:,3:6]
            for i in range(3,len(rhand_centric_indexes), 3):
                take.motion.data[:,rhand_centric_indexes[i:i+3]] = take.motion.data[:,rhand_centric_indexes[i:i+3]] - take.motion.data[:,69:72]
            for i in range(3,len(lhand_centric_indexes), 3):
                take.motion.data[:,lhand_centric_indexes[i:i+3]] = take.motion.data[:,lhand_centric_indexes[i:i+3]] - take.motion.data[:,201:204]


    def createLabels(self):
        self.classes = 12
        for id in range(1,3):
            sum = 6 if id == 2 else 0
            for take in self.getId(id):
                take.takelabel = np.zeros(self.classes)
                if take.p==1:
                    if take.e==1:
                        take.takelabel[0+sum] = 1
                    elif take.e==2:
                        take.takelabel[1+sum] = 1
                    elif take.e==3:
                        take.takelabel[2+sum] = 1
                elif take.p==2:
                    if take.e==1:
                        take.takelabel[3+sum] = 1
                    elif take.e==2:
                        take.takelabel[4+sum] = 1
                    elif take.e==3:
                        take.takelabel[5+sum] = 1
                    
        # Remove takes without labels
        self.takes = [take for take in self.takes if take.takelabel.any()]


    def getTake(self, name):
        info = name.split('_')
        try:
            info_id, info_p, info_e, info_f = int(info[0][2:]), int(info[1][1:]), int(info[2][1:]), int(info[3][1:])
            for take in self.takes:
                if take.id == info_id:
                    if take.p == info_p:
                        if take.e == info_e:
                            if take.f == info_f:
                                return take
        except:
            print("Please use the format \"idxx_pxx_exx_fxx\".")
        

    def getId(self, info_id):
        assert type(info_id)==int
        for take in self.takes:
            if take.id == info_id:
                yield take

    def getPart(self, info_p, info_id=None):
        if info_id:
            assert type(info_id)==int and type(info_p)==int
            for take in self.takes:
                if take.id == info_id and take.p == info_p:
                    yield take
        else:
            assert type(info_p)==int
            for take in self.takes:
                if take.p == info_p:
                    yield take


    def getStyle(self, info_id, info_p, info_e):
        assert type(info_id)==int and type(info_p)==int and type(info_e)==int
        for take in self.takes:
            if take.id == info_id:
                if take.p == info_p:
                    if take.e == info_e:
                        yield take

    

    def old__getitem__(self, index):
        take, index = self.index2take(index)
        sample_range = self.samples_index_per_take[take][index]
        return self.takes[take].motion.data[sample_range[0]:sample_range[1],:], self.takes[take].takelabel
    
    
class Take:
    def __init__(self, 
                 motionname, 
                 audioname,
                 npy_motion_path,
                 npy_audio_path):
        # Get name
        self.ref = motionname.split('.')[0]
        # Get info from name
        self._register()
        # Get data
        self.motion = Motion(path.join(npy_motion_path, motionname), self)
        self.audio = Audio(path.join(npy_audio_path, audioname), self)
        self.takelabel=None

    def _register(self):
        info = self.ref.split('_')
        self.id, self.p, self.e, self.f = int(info[0][2:]), int(info[1][1:]), int(info[2][1:]), int(info[3][1:])
        self.detail()

    def detail(self):
        print("id{:02} p{:02} e{:02} f{:02}".format(self.id, self.p, self.e, self.f))

class Audio:
    def __init__(self, path, take):
        self.path = path
        self.data = np.load(self.path)
        self.take = take

class Motion:
    def __init__(self, path, take):
        self.path = path
        self.data = np.load(self.path)
        self.take = take

def bvh2npyconverter(overwrite = False):
    npypath = os.path.join(MOTIONPATH, 'npy')
    # Creates a path to npy files inside the dataset file
    # if overwrite = False look for a available dir such as "npy_3"
    # if overwrite = True simply choose dir "npy"
    if os.path.exists(npypath):
        if not overwrite:
            i = 2
            while os.path.exists(npypath+'_'+str(i)):
                i += 1
            npypath = npypath+'_'+str(i)
            os.makedirs(npypath)
    else:
        os.makedirs(npypath)

    bvhfiles = os.listdir(BVHPATH)
    bvhfiles.sort()
    for bvhfile in bvhfiles:
        anim = bvhsdk.ReadFile(os.path.join(BVHPATH, bvhfile))
        npyfile = np.empty(shape=(anim.frames, 6*len(anim.getlistofjoints())))
        for i, joint in enumerate(anim.getlistofjoints()):
            npyfile[:, i*6:i*6+3] = joint.rotation
            for frame in range(anim.frames):
                npyfile[frame, i*6+3:i*6+6] = joint.getPosition(frame)

        np.save(os.path.join(npypath, bvhfile)[:-4], npyfile)
        print('%s done.' % bvhfile)


    #Save joint names (of the last bvh file)
    joint_names = []
    for joint in anim.getlistofjoints():
        joint_names.append(joint.name)
    with open(os.path.join(npypath, "joints"), "wb") as j:
        pickle.dump(joint_names, j)

def wav2npyconverter(overwrite = False, fps=120):
    npypath = os.path.join(AUDIOPATH, 'npy')
    # Creates a path to npy files inside the dataset file
    # if overwrite = False look for a available dir such as "npy_3"
    # if overwrite = True simply choose dir "npy"
    if os.path.exists(npypath):
        if not overwrite:
            i = 2
            while os.path.exists(npypath+'_'+str(i)):
                i += 1
            npypath = npypath+'_'+str(i)
            os.makedirs(npypath)
    else:
        os.makedirs(npypath)

    wavfiles = os.listdir(WAVPATH)
    wavfiles.sort()
    for wavfile in wavfiles:
        print(wavfile)
        fs,signal = wav.read(os.path.join(WAVPATH, wavfile))
        signal_ = signal.astype(float)/math.pow(2,15)
        assert fs%fps == 0
        hop_len=int(fs/fps)
        n_fft=int(fs*0.13)
        C = librosa.feature.melspectrogram(y=signal_, sr=fs, n_fft=2048, hop_length=hop_len, n_mels=27, fmin=0.0, fmax=8000)
        np.save(os.path.join(npypath, wavfile)[:-4], C)
