import numpy as np


def create_4(config):
    pass

def create_3(config):
    pass

def create_pyr_pv(config, seed=1234):
    np.random.seed(seed)
    rng = np.random.default_rng()
    pee = config["pee"]
    pie = config["pie"]
    pii = config["pii"]
    pei = config["pei"]

    Ne = int(config["N"] * config["pyr_ratio"])
    Ni = int(config["N"] * config["pv_ratio"])

    Kee = pee * Ne # expected  of connections (for e to e)'
    Kie = pie * Ne
    Kei = pei * Ni
    Kii = pii * Ni

    # connection strengths default values
    jee = config["jee"]
    jie = config["jie"]
    jei = config["jei"]
    jii = config["jii"]

    # external drives

    je0 = config["je0"]
    ji0 = config["ji0"]

    # scaling the true recurrent weights by 1 / sqrt {K}
    Jee = jee / np.sqrt(Kee)
    Jie = jie / np.sqrt(Kie)
    Jei = jei / np.sqrt(Kei)
    Jii = jii / np.sqrt(Kii)

    #scaling the feedforward weights by sqrt  {K}
    Je0 = je0 * np.sqrt(Kee)
    Ji0 = ji0 * np.sqrt(Kie)

    # membrane dynamics

    Vt = 1 # threshold
    Vr = 0 # reset
    tau = 10 # this is the membrane timescale of the neuron

    # simulation details
    dt = tau / 500 # this should be accurate enough
    T = 2000 # total time
    maxspk = 100000 # pre - allocate space for the spike trains

    # connection matrices

    # E to E
    cEE = np.zeros((Ne, Ne)) # storing E to E connections
    cEE_t = rng.random((Ne, Ne)) # random matrix
    cEE[cEE_t < pee] = 1 # adjacency matrix(0 if no connection, 1 if connection)

    # E to I

    cIE = np.zeros((Ne, Ni))
    cIE_t = rng.random((Ne, Ni))
    cIE[cIE_t < pie] = 1

    # I to E

    cEI = np.zeros((Ni, Ne))
    cEI_t = rng.random((Ni, Ne))
    cEI[cEI_t < pei] = 1

    # I to

    cII = np.zeros((Ni, Ni))
    cII_t = rng.random((Ni, Ni))
    cII[cII_t < pii] = 1

    # intial conditions
    Ve = rng.random(Ne)
    Vi = rng.random(Ni)
    Je0_vec = Je0 * np.ones(Ne)
    Ji0_vec = Ji0 * np.ones(Ni)

    # spiketime arrays; use a pre - allocatedsize
    spktime_e = np.zeros(maxspk)
    spkindex_e = np.zeros(maxspk)
    spktime_i = np.zeros(maxspk)
    spkindex_i = np.zeros(maxspk)

    # intialize the spike time counter
    counte = 0
    counti = 0

    # time loop

    for t in np.linspace(0, T, int(T/dt)):

        # zero the interactions from the last step
        kickee = np.zeros(Ne)
        kickei = np.zeros(Ne)
        kickie = np.zeros(Ni)
        kickii = np.zeros(Ni)

        index_spke = np.where(Ve >= Vt)[0]  # find e spikers

        if len(index_spke) > 0:
            spktime_e[counte:counte + len(index_spke)] = t  # update spktime array
            spkindex_e[counte:counte + len(index_spke)] = index_spke  # update spkindex array
            Ve[index_spke] = Vr  # reset the spikers
            counte = counte + len(index_spke)  # update the counter

        index_spki = np.where(Vi >= Vt)[0]  # find i spikers

        if len(index_spki) > 0:
            spktime_i[counti:counti + len(index_spki)] = t
            spkindex_i[counti:counti + len(index_spki)] = index_spki
            Vi[index_spki] = Vr
            counti = counti + len(index_spki)

        # update e synaptic kick vectors

        for j in range(1, len(index_spke)):  # loop over spikers

            kickee_index = np.where(cEE[index_spke[j], :] > 0)[0]  # find the post-synaptic targets of spiker j
            kickee[kickee_index] = kickee[kickee_index] + 1  # update the post-synaptic targets kick

            kickie_index = np.where(cIE[index_spke[j], :] > 0)[0]
            kickie[kickie_index] = kickie[kickie_index] + 1

        # update i synaptic kick vectors

        for j in range(1, len(index_spki)):
            kickei_index = np.where(cEI[index_spki[j], :] > 0)[0]
            kickei[kickei_index] = kickei[kickei_index] + 1

            kickii_index = np.where(cII[index_spki[j], :] > 0)[0]
            kickii[kickii_index] = kickii[kickii_index] + 1

        # kick the Vs

        Ve = Ve + Jee * kickee - Jei * kickei # kick the e neurons Vm
        Vi = Vi + Jie * kickie - Jii * kickii # kick the i neurons Vm

        # integrate the Vs

        Ve = Ve + dt / tau * (-Ve + Je0_vec) # e membrane integration
        Vi = Vi + dt / tau * (-Vi + Ji0_vec) # i membrane integration

        result = dict(spkindex_i=spkindex_i,
                      spktime_i=spktime_i,
                      counti=counti,
                      spkindex_e=spkindex_e,
                      spktime_e=spktime_e,
                      counte=counte)
    return result


def create_pyr_pv(config, seed=1234):
    np.random.seed(seed)
    rng = np.random.default_rng()
    pee = config["pee"]
    pie = config["pie"]
    pii = config["pii"]
    pei = config["pei"]

    Ne = int(config["N"] * config["pyr_ratio"])
    Ni = int(config["N"] * config["pv_ratio"])

    Kee = pee * Ne # expected  of connections (for e to e)'
    Kie = pie * Ne
    Kei = pei * Ni
    Kii = pii * Ni

    # connection strengths default values
    jee = config["jee"]
    jie = config["jie"]
    jei = config["jei"]
    jii = config["jii"]

    # external drives

    je0 = config["je0"]
    ji0 = config["ji0"]

    # scaling the true recurrent weights by 1 / sqrt {K}
    Jee = jee / np.sqrt(Kee)
    Jie = jie / np.sqrt(Kie)
    Jei = jei / np.sqrt(Kei)
    Jii = jii / np.sqrt(Kii)

    #scaling the feedforward weights by sqrt  {K}
    Je0 = je0 * np.sqrt(Kee)
    Ji0 = ji0 * np.sqrt(Kie)

    # membrane dynamics

    Vt = 1 # threshold
    Vr = 0 # reset
    tau = 10 # this is the membrane timescale of the neuron

    # simulation details
    dt = tau / 500 # this should be accurate enough
    T = 2000 # total time
    maxspk = 100000 # pre - allocate space for the spike trains

    # connection matrices

    # E to E
    cEE = np.zeros((Ne, Ne)) # storing E to E connections
    cEE_t = rng.random((Ne, Ne)) # random matrix
    cEE[cEE_t < pee] = 1 # adjacency matrix(0 if no connection, 1 if connection)

    # E to I

    cIE = np.zeros((Ne, Ni))
    cIE_t = rng.random((Ne, Ni))
    cIE[cIE_t < pie] = 1

    # I to E

    cEI = np.zeros((Ni, Ne))
    cEI_t = rng.random((Ni, Ne))
    cEI[cEI_t < pei] = 1

    # I to

    cII = np.zeros((Ni, Ni))
    cII_t = rng.random((Ni, Ni))
    cII[cII_t < pii] = 1

    # intial conditions
    Ve = rng.random(Ne)
    Vi = rng.random(Ni)
    Je0_vec = Je0 * np.ones(Ne)
    Ji0_vec = Ji0 * np.ones(Ni)

    # spiketime arrays; use a pre - allocatedsize
    spktime_e = np.zeros(maxspk)
    spkindex_e = np.zeros(maxspk)
    spktime_i = np.zeros(maxspk)
    spkindex_i = np.zeros(maxspk)

    # intialize the spike time counter
    counte = 0
    counti = 0

    # time loop

    for t in np.linspace(0, T, int(T/dt)):

        # zero the interactions from the last step
        kickee = np.zeros(Ne)
        kickei = np.zeros(Ne)
        kickie = np.zeros(Ni)
        kickii = np.zeros(Ni)

        index_spke = np.where(Ve >= Vt)[0]  # find e spikers

        if len(index_spke) > 0:
            spktime_e[counte:counte + len(index_spke)] = t  # update spktime array
            spkindex_e[counte:counte + len(index_spke)] = index_spke  # update spkindex array
            Ve[index_spke] = Vr  # reset the spikers
            counte = counte + len(index_spke)  # update the counter

        index_spki = np.where(Vi >= Vt)[0]  # find i spikers

        if len(index_spki) > 0:
            spktime_i[counti:counti + len(index_spki)] = t
            spkindex_i[counti:counti + len(index_spki)] = index_spki
            Vi[index_spki] = Vr
            counti = counti + len(index_spki)

        # update e synaptic kick vectors

        for j in range(1, len(index_spke)):  # loop over spikers

            kickee_index = np.where(cEE[index_spke[j], :] > 0)[0]  # find the post-synaptic targets of spiker j
            kickee[kickee_index] = kickee[kickee_index] + 1  # update the post-synaptic targets kick

            kickie_index = np.where(cIE[index_spke[j], :] > 0)[0]
            kickie[kickie_index] = kickie[kickie_index] + 1

        # update i synaptic kick vectors

        for j in range(1, len(index_spki)):
            kickei_index = np.where(cEI[index_spki[j], :] > 0)[0]
            kickei[kickei_index] = kickei[kickei_index] + 1

            kickii_index = np.where(cII[index_spki[j], :] > 0)[0]
            kickii[kickii_index] = kickii[kickii_index] + 1

        # kick the Vs

        Ve = Ve + Jee * kickee - Jei * kickei # kick the e neurons Vm
        Vi = Vi + Jie * kickie - Jii * kickii # kick the i neurons Vm

        # integrate the Vs

        Ve = Ve + dt / tau * (-Ve + Je0_vec) # e membrane integration
        Vi = Vi + dt / tau * (-Vi + Ji0_vec) # i membrane integration

        result = dict(spkindex_i=spkindex_i,
                      spktime_i=spktime_i,
                      counti=counti,
                      spkindex_e=spkindex_e,
                      spktime_e=spktime_e,
                      counte=counte)
    return result

if __name__ == "__main__":

    from config import networks

    network_config = networks["Pyr-PV"]

    res = create_pyr_pv(config=network_config)

    from plot import plot_raster

    plot_raster(spk_idx=res["spkindex_i"], spk_time=res["spktime_i"])










