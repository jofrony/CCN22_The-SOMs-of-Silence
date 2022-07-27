import elephant
import neo
from scipy import io
import scipy
import numpy as np
import random
import quantities as pq

def rate_per_neuron(idx, time, duration):

    cells = np.unique(idx)
    rates = list()

    for c in cells[1:]:
        spk = np.where(idx.T[0] == c)[0]
        
        r = (len(spk) * 1e3) / duration
        rates.append(r)

    return rates

def rate(count, num, duration):
    return (count * 1e3) / (duration / num)


def corrcoef_analysis(neuron_sample_size, binsize):
    spktime_e = scipy.io.loadmat("test_e_spktime.mat")
    spkidx_e = scipy.io.loadmat("test_e_spkindx.mat")
    neuron_uq = np.unique(spkidx_e["spkindex_e"].T[0])[1:]
    num_neurons = neuron_sample_size
    binsize = binsize * pq.ms
    sample_neurons = random.sample(list(neuron_uq), num_neurons)
    neo_spiketrains = list()

    for s in sample_neurons:
        neuron_idx = np.where(spkidx_e["spkindex_e"].T[0] == s)[0]

        spk = np.take(spktime_e['spktime_e'].T[0], neuron_idx)

        neo_spk = neo.SpikeTrain(spk * pq.ms, t_stop=2000 * pq.s)

        neo_spiketrains.append(neo_spk)

        binned_spk = elephant.conversion.BinnedSpikeTrain(neo_spiketrains, binsize=binsize)

    corr = elephant.spike_train_correlation.correlation_coefficient(binned_spk)

    return corr


if __name__ == "__main__":

    import numpy as np

    bins_size = np.arange(10, 200, 10)
    sample_size = 100

    mean_corr = list()
    for b in bins_size:
        corr = corrcoef_analysis(neuron_sample_size=sample_size, binsize=b)
        mean_corr.append(np.mean(corr.ravel()))
