import cv2
import numpy as np

def prepare_kintree():

    kintree_flat = np.array([[4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,12,13,14,16,17,18,19,20,21],
                                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]])

    kintree = {v[1]: v[0] for v in kintree_flat[:,1:].T}
    #kintree = {}
    #for i in range(24):
    #    parent_idx = i
    #    kintree[i] = []    
    #    while parent_idx < 25:
    #        kintree[i] = [parent_idx] + kintree[i]
    #        parent_idx = kintree_flat[0, kintree_flat[1, parent_idx]]
    return kintree

def aar_to_quaternion(latent_inp, kintree=None):

    len_input = latent_inp.shape[0]
    assert len_input % 3 == 0
    assert len(latent_inp.shape) == 1

    latent_out = np.zeros((4*len_input/3,))

    for i, m in enumerate(range(0,len_input,3)):
        aar = latent_inp[m:m+3]
        angle = np.linalg.norm(aar) # % (2*np.pi)
        quat = np.concatenate(([np.cos(angle/2.)], np.sin(angle/2.) * aar / angle))
        if quat[0] < 0:
            quat = -quat
        latent_out[i*4:(i+1)*4] = quat / np.linalg.norm(quat) # [qw, qx, qy, qz]

    return latent_out.astype(np.float32)

def rotmat_to_quaternion(latent_inp, kintree=None):

    latent_inp = rotmat_to_aar(latent_inp, kintree)
    latent_out = aar_to_quaternion(latent_inp, kintree)

    return latent_out.astype(np.float32)

def aar_to_rotmat(latent_inp, kintree=None):

    len_input = latent_inp.shape[0]
    assert len_input % 3 == 0
    assert len(latent_inp.shape) == 1

    latent_out = np.zeros((len_input*3,))

    rot_matrices_rel = [[] for i in range(len_input/3)]
    if kintree is not None:
        assert len_input == 72, "for this mode, we assume all joint angles are included"
        rot_matrices_abs = [np.identity(3) for i in range(len_input/3)]
        rot_matrices_abs[0] = cv2.Rodrigues(latent_inp[:3])[0]

    for i, m in enumerate(range(0,len_input,3)):
        rot_matrices_rel[i] = cv2.Rodrigues(latent_inp[m:m+3])[0]

    latent_out[:9] = rot_matrices_rel[0].flatten()
    for i, m in enumerate(range(9,len_input*3,9), 1):
        if kintree is not None:
            p = kintree[i]
            rot_matrices_abs[i] = rot_matrices_abs[p].dot(rot_matrices_rel[i])
            latent_out[m:m+9] = rot_matrices_abs[i].flatten()
        else:
            latent_out[m:m+9] = rot_matrices_rel[i].flatten()

    return latent_out

def rotmat_to_aar(latent_inp, kintree=None):

    len_input = latent_inp.shape[0]
    assert len_input % 9 == 0, latent_inp.shape
    assert len(latent_inp.shape) == 1

    latent_out = np.zeros((len_input/3,))

    if kintree is not None:
        assert len_input == 216, "for this mode, we assume all joint angles are included"
        rot_mats_abs = [[] for i in range(len_input/9)]
        for i, m in enumerate(range(0,len_input,9)):
            rot_mats_abs[i] = latent_inp[m:m+9].reshape(3,3)

        latent_out[:3] = np.squeeze(cv2.Rodrigues(rot_mats_abs[0])[0])
        for j in range(1, len_input/9):
            latent_out[j*3:(j+1)*3] = cv2.Rodrigues(rot_mats_abs[kintree[j]].T.dot(rot_mats_abs[j]))[0].flatten()
    else:
        for x, i in enumerate(range(0,len_input,9)):
            latent_out[x*3:(x+1)*3] = cv2.Rodrigues(np.reshape(latent_inp[i:i+9], (3,3)))[0].flatten()

    return latent_out

def aar_to_rotmat_old(latent_inp, kintree=None):

    latent_out = np.zeros((227,))
    latent_out[:10] = latent_inp[:10]
    latent_out[226] = latent_inp[82]

    rot_matrices_rel = [[] for i in range(24)]
    if kintree is not None:
        rot_matrices_abs = [np.identity(3) for i in range(24)]

    for i, m in enumerate(range(10,82,3)):
        rot_matrices_rel[i] = cv2.Rodrigues(latent_inp[m:m+3])[0]

    for i, m in enumerate(range(10,226,9)):
        if kintree is not None:
            for p in kintree[i]:
                rot_matrices_abs[i] = rot_matrices_rel[p].dot(rot_matrices_abs[i])
            latent_out[m:m+9] = rot_matrices_abs[i].flatten()
        else:
            latent_out[m:m+9] = rot_matrices_rel[i].flatten()

    return latent_out
