import numpy as np
from utils import make_connectivity_matrix

def create_pyr_pv_som(config, seed=1234):
    np.random.seed(seed)
    rng = np.random.default_rng()
    p_e_e = config["p_e_e"]
    p_pv_e = config["p_pv_e"]
    p_pv_pv = config["p_pv_pv"]
    p_e_pv = config["p_e_pv"]

    p_som_e = config["p_som_e"]
    p_som_pv = config["p_som_pv"]

    N_e = int(config["N"] * config["pyr_ratio"])
    N_pv = int(config["N"] * config["pv_ratio"])
    N_som = int(config["N"] * config["som_ratio"])

    K_e_e = p_e_e * N_e # expected  of connections (for e to e)'
    K_pv_e = p_pv_e * N_e
    K_e_pv = p_e_pv * N_pv
    K_pv_pv = p_pv_pv * N_pv

    K_som_e = p_som_e * N_som
    K_som_pv = p_som_pv * N_som

    # connection streng ths default values
    j_e_e = config["j_e_e"]
    j_pv_e = config["j_pv_e"]
    j_e_pv = config["j_e_pv"]
    j_pv_pv = config["j_pv_pv"]

    j_som_e = config["j_som_e"]
    j_som_pv = config["j_som_pv"]

    # external drives

    j_e_0 = config["j_e_0"]
    j_pv_0 = config["j_pv_0"]
    j_som_0 = config["j_som_0"]

    # scaling the true recurrent weights by 1 / sqrt {K}
    J_e_e = j_e_e / np.sqrt(K_e_e)
    J_pv_e = j_pv_e / np.sqrt(K_pv_e)
    J_e_pv = j_e_pv / np.sqrt(K_e_pv)
    J_pv_pv = j_pv_pv / np.sqrt(K_pv_pv)

    J_som_e = j_som_e / np.sqrt(K_som_e)
    J_som_pv = j_som_pv / np.sqrt(K_som_pv)

    #scaling the feedforward weights by sqrt  {K}
    J_e_0 = j_e_0 * np.sqrt(K_e_e)
    J_pv_0 = j_pv_0 * np.sqrt(K_pv_e)

    J_som_0 = j_som_0 * np.sqrt(K_som_e)

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

    c_e_e = make_connectivity_matrix(N_e, N_e, p_e_e, rng)

    # E to PV

    c_pv_e = make_connectivity_matrix(N_e, N_pv, p_pv_e, rng)

    # PV to E

    c_e_pv = make_connectivity_matrix(N_pv, N_e, p_e_pv, rng)

    # PV to PV

    c_pv_pv = make_connectivity_matrix(N_pv, N_pv, p_pv_pv, rng)

    # SOM to E

    c_som_e = make_connectivity_matrix(N_e, N_som, p_som_e, rng)

    # SOM to PV

    c_som_pv = make_connectivity_matrix(N_pv, N_som, p_som_pv, rng)

    # intial conditions
    V_e = rng.random(N_e)
    V_pv = rng.random(N_pv)
    V_som = rng.random(N_som)
    Je0_vec = J_e_0 * np.ones(N_e)
    Jpv0_vec = J_pv_0 * np.ones(N_pv)
    Jsom0_vec = J_som_0 * np.ones(N_som)

    # spiketime arrays; use a pre - allocatedsize
    spktime_e = np.zeros(maxspk)
    spkindex_e = np.zeros(maxspk)

    spktime_pv = np.zeros(maxspk)
    spkindex_pv = np.zeros(maxspk)

    spktime_som = np.zeros(maxspk)
    spkindex_som = np.zeros(maxspk)

    # intialize the spike time counter
    counte = 0
    countpv = 0
    countsom = 0

    # time loop

    for t in np.linspace(0, T, int(T/dt)):

        # zero the interactions from the last step
        kickee = np.zeros(N_e)
        kickepv = np.zeros(N_e)
        kickpve = np.zeros(N_pv)
        kickpvpv = np.zeros(N_pv)

        kicksome = np.zeros(N_e)
        kicksompv = np.zeros(N_pv)

        index_spke = np.where(V_e >= Vt)[0]  # find e spikers

        if len(index_spke) > 0:
            spktime_e[counte:counte + len(index_spke)] = t  # update spktime array
            spkindex_e[counte:counte + len(index_spke)] = index_spke  # update spkindex array
            V_e[index_spke] = Vr  # reset the spikers
            counte = counte + len(index_spke)  # update the counter

        index_spkpv = np.where(V_pv >= Vt)[0]  # find i spikers

        if len(index_spkpv) > 0:
            spktime_pv[countpv:countpv + len(index_spkpv)] = t
            spkindex_pv[countpv:countpv + len(index_spkpv)] = index_spkpv
            V_pv[index_spkpv] = Vr
            countpv = countpv + len(index_spkpv)

        index_spksom = np.where(V_som >= Vt)[0]  # find i spikers

        if len(index_spksom) > 0:
            spktime_som[countsom:countsom + len(index_spksom)] = t
            spkindex_som[countsom:countsom + len(index_spksom)] = index_spksom
            V_som[index_spksom] = Vr
            countsom = countsom + len(index_spksom)

        # update e synaptic kick vectors

        for j in range(1, len(index_spke)):  # loop over spikers

            kickee_index = np.where(c_e_e[index_spke[j], :] > 0)[0]  # find the post-synaptic targets of spiker j
            kickee[kickee_index] = kickee[kickee_index] + 1  # update the post-synaptic targets kick

            kickpve_index = np.where(c_pv_e[index_spke[j], :] > 0)[0]
            kickpve[kickpve_index] = kickpve[kickpve_index] + 1

        # update i synaptic kick vectors

        for j in range(1, len(index_spkpv)):
            kickepv_index = np.where(c_e_pv[index_spkpv[j], :] > 0)[0]
            kickepv[kickepv_index] = kickepv[kickepv_index] + 1

            kickpvpv_index = np.where(c_pv_pv[index_spkpv[j], :] > 0)[0]
            kickpvpv[kickpvpv_index] = kickpvpv[kickpvpv_index] + 1

        for j in range(1, len(index_spksom)):
            kicksome_index = np.where(c_som_e[index_spksom[j], :] > 0)[0]
            kicksome[kicksome_index] = kicksome[kicksome_index] + 1

            kicksompv_index = np.where(c_som_pv[index_spksom[j], :] > 0)[0]
            kicksompv[kicksompv_index] = kicksompv[kicksompv_index] + 1

        # kick the Vs

        V_e = V_e + J_e_e * kickee - J_pv_e * kickepv - J_som_e * kicksome # kick the e neurons Vm
        V_pv = V_pv + J_e_pv * kickpve - J_pv_pv * kickpvpv - J_som_pv * kicksompv # kick the i neurons Vm

        # integrate the Vs

        V_e = V_e + dt / tau * (-V_e + Je0_vec)  # e membrane integration
        V_pv = V_pv + dt / tau * (-V_pv + Jpv0_vec) # i membrane integration
        V_som = V_som + dt / tau * (-V_som + Jsom0_vec)  # i membrane integration


        result = dict(spkindex_pv=spkindex_pv,
                      spktime_pv=spktime_pv,
                      spkindex_som=spkindex_som,
                      spktime_som=spktime_som,
                      countpv=countpv,
                      spkindex_e=spkindex_e,
                      spktime_e=spktime_e,
                      counte=counte)
    return result

if __name__ == "__main__":

    from config import networks

    network_config = networks["SOM<->Pyr-PV<-SOM"]

    res = create_pyr_pv_som(config=network_config)

    from plot import plot_raster

    plot_raster(spk_idx=res["spkindex_e"], spk_time=res["spktime_e"])

    plot_raster(spk_idx=res["spkindex_pv"], spk_time=res["spktime_pv"])

    plot_raster(spk_idx=res["spkindex_som"], spk_time=res["spktime_som"])

