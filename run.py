import hashlib
import math
import os

import pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import numpy.random as rnd


def createKey(n: int, Q: float):
    AliceKey = rnd.randint(0, 2, size=n)
    error_line = np.array([1 if rnd.random() < Q else 0 for _ in range(n)])
    BobKey = AliceKey ^ error_line
    return AliceKey, BobKey


def list_to_matrix(input_list, k, n):
    if len(input_list) < k * n:
        while len(input_list) != k * n:
            input_list.append(0)
    matrix = np.array(input_list).reshape(k, n)
    return matrix


class TreeParityMachine:
    def __init__(self, K, N, L, weights):
        self.K = K
        self.N = N
        self.L = L
        self.weights = weights

    def update(self, X, tau_Partner):
        tau = self.tau
        sigma = self.sigma
        weights = self.weights
        L = self.L
        if tau == tau_Partner:
            self.hebbian(weights, X, sigma, tau, tau_Partner, L)

    def compute_output(self, X):
        X = X.reshape([self.K, self.N])
        sigma = np.where(np.sum(X * self.weights, axis=1) > 0, 1, -1)
        tau = np.prod(sigma)
        self.X = X
        self.sigma = sigma
        self.tau = tau
        return tau

    def hebbian(self, weights, X, sigma, tau_A, tau_B, l):
        mask = (sigma == tau_A).astype(weights.dtype).reshape(-1, 1)
        weights += X * tau_A * mask
        np.clip(weights, -l, l, out=weights)


def binary_to_decimal(binary_array, rangeBits):
    weightList = []
    minus = 0
    for k in range(rangeBits):
        minus += 2 ** k
    minus = minus // 2
    minus += 1
    for i in range(0, len(binary_array), rangeBits):
        torture = ''
        for j in range(rangeBits):
            torture += str(binary_array[i + j])
        a = int(torture, 2)
        a -= minus
        weightList.append(a)
    return weightList


def disclose_one_bit_to_eve(alice_key, eve_key_estimate, eve_machines, Bits):
    differing_positions = np.flatnonzero(alice_key != eve_key_estimate)
    if differing_positions.size == 0:
        return False

    bit_position = int(rnd.choice(differing_positions))
    weight_position = bit_position // Bits
    block_start = weight_position * Bits
    block_end = block_start + Bits

    old_block = eve_key_estimate[block_start:block_end].tolist()
    old_weight = binary_to_decimal(old_block, Bits)[0]

    eve_key_estimate[bit_position] = alice_key[bit_position]

    new_block = eve_key_estimate[block_start:block_end].tolist()
    new_weight = binary_to_decimal(new_block, Bits)[0]
    weight_delta = new_weight - old_weight

    for eve_machine in eve_machines:
        eve_machine.weights.flat[weight_position] = np.clip(
            eve_machine.weights.flat[weight_position] + weight_delta,
            -eve_machine.L,
            eve_machine.L
        )

    return True


def calculate_correlation(weights_a, weights_b):
    a = np.asarray(weights_a, dtype=float).flatten()
    b = np.asarray(weights_b, dtype=float).flatten()

    if a.size != b.size:
        raise ValueError("Наборы весов должны иметь одинаковый размер.")

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def weights_to_binary(weights, Bits):
    L = 2 ** (Bits - 1)
    values = np.asarray(weights, dtype=int).flatten() + L

    if np.any(values < 0) or np.any(values > 2 * L):
        raise ValueError("Вес выходит за допустимый диапазон")

    width = Bits + 1
    binary_string = np.fromiter(
        (
            int(bit)
            for value in values
            for bit in format(int(value), f"0{width}b")
        ),
        dtype=np.uint8,
        count=len(values) * width
    )
    return binary_string


def calculate_mutual_information(binary_a, binary_b):
    a = np.asarray(binary_a, dtype=int).flatten()
    b = np.asarray(binary_b, dtype=int).flatten()

    if a.size != b.size:
        raise ValueError("Длины бинарных последовательностей не совпадают")

    if a.size == 0:
        return np.nan

    joint_counts = np.bincount(
        2 * a + b,
        minlength=4
    ).reshape(2, 2)

    joint = joint_counts / a.size
    px = np.sum(joint, axis=1)
    py = np.sum(joint, axis=0)
    expected = px[:, None] * py[None, :]
    mask = joint > 0
    mi = np.sum(joint[mask] * np.log2(joint[mask] / expected[mask]))

    return float(np.clip(mi, 0.0, 1.0))


def get_parameter_sets_for_key_length(target_key_length=8192, bits_min=1, bits_max=13):
    parameter_sets = []
    for Bits in range(bits_min, bits_max + 1):
        if target_key_length % Bits != 0:
            continue
        product_kn = target_key_length // Bits
        for K in range(1, product_kn + 1):
            if product_kn % K == 0:
                N = product_kn // K
                if K <= N:
                    parameter_sets.append([K, N, Bits])
    return parameter_sets


def _active_attack_names(eve_attack):
    return [eve_attack]


def _attack_suffix(attack):
    return {
        "naive": "Naive",
        "geometric": "Geometric",
        "both": "Both",
    }[attack]


def _update_eve(TPM_E, X, out_A, out_B, out_E, attack):
    if attack == "naive":
        if out_A == out_E:
            TPM_E.update(X, out_A)

    elif attack == "geometric":
        if out_A == out_B:
            if out_E != out_A:
                local_fields = np.sum(X * TPM_E.weights, axis=1)
                neuron = np.argmin(np.abs(local_fields))
                TPM_E.sigma[neuron] *= -1
                TPM_E.tau = np.prod(TPM_E.sigma)
            TPM_E.update(X, out_A)

    else:
        if out_A == out_E:
            TPM_E.update(X, out_A)
        elif out_A == out_B:
            local_fields = np.sum(X * TPM_E.weights, axis=1)
            neuron = np.argmin(np.abs(local_fields))
            TPM_E.sigma[neuron] *= -1
            TPM_E.tau = np.prod(TPM_E.sigma)
            TPM_E.update(X, out_A)


def _measure_chunk(args):
    (K, N, Bits, M, measurements_chunk, max_iterations, Q, seed,
     eve_attack, leakage_enabled) = args

    rnd.seed(seed)

    n_local = K * N * Bits
    L_local = (2 ** Bits) // 2

    failures = 0
    sync_iterations_sum = 0
    disclosed_information_sum = 0.0
    fact_sync_iterations_sum = 0
    leaked_bits_count = 0

    active_attacks = _active_attack_names(eve_attack)
    eve_stats = {
        attack: {
            "correlation_sum": np.zeros(max_iterations, dtype=np.float64),
            "correlation_count": np.zeros(max_iterations, dtype=np.int64),
            "mutual_information_sum": np.zeros(max_iterations, dtype=np.float64),
            "mutual_information_count": np.zeros(max_iterations, dtype=np.int64),
            "first_sync_count": 0,
            "first_sync_iteration_sum": 0,
            "faster_than_bob_count": 0,
        }
        for attack in active_attacks
    }

    correlation_ab_sum = np.zeros(max_iterations, dtype=np.float64)
    correlation_ab_count = np.zeros(max_iterations, dtype=np.int64)
    mutual_information_ab_sum = np.zeros(max_iterations, dtype=np.float64)
    mutual_information_ab_count = np.zeros(max_iterations, dtype=np.int64)

    for _ in range(measurements_chunk):
        keySlice_A, keySlice_B = createKey(n_local, Q)
        keySlice_E = rnd.randint(0, 2, size=n_local)
        eve_key_estimate = np.array(keySlice_E, copy=True)

        weights_A = binary_to_decimal(keySlice_A, Bits)
        weights_B = binary_to_decimal(keySlice_B, Bits)
        weights_E = binary_to_decimal(keySlice_E, Bits)

        arrayWeights_A = list_to_matrix(weights_A, K, N)
        arrayWeights_B = list_to_matrix(weights_B, K, N)
        arrayWeights_E = list_to_matrix(weights_E, K, N)

        TPM_A = TreeParityMachine(K, N, L_local, arrayWeights_A)
        TPM_B = TreeParityMachine(K, N, L_local, arrayWeights_B)
        TPM_E = TreeParityMachine(
            K, N, L_local, np.array(arrayWeights_E, copy=True)
        )

        first_sync_ab = None
        first_sync_eve = {attack: None for attack in active_attacks}

        for iter_ in range(1, max_iterations + 1):
            if Bits > 1:
                vals = np.r_[np.arange(-M, 0), np.arange(1, M + 1)]
                X = np.random.choice(vals, size=(K, N))
            else:
                X = np.random.randint(-1, 2, [K, N])

            out_A = TPM_A.compute_output(X)
            out_B = TPM_B.compute_output(X)
            out_E = TPM_E.compute_output(X)

            if out_A == out_B:
                TPM_A.update(X, out_B)
                TPM_B.update(X, out_A)

            if leakage_enabled:
                if disclose_one_bit_to_eve(
                    keySlice_A, eve_key_estimate, [TPM_E], Bits
                ):
                    leaked_bits_count += 1
                    out_E = TPM_E.compute_output(X)

            _update_eve(TPM_E, X, out_A, out_B, out_E, eve_attack)

            weights_equal_ab = np.array_equal(
                TPM_A.weights.flatten(), TPM_B.weights.flatten()
            )

            if first_sync_ab is None and weights_equal_ab:
                first_sync_ab = iter_
                fact_sync_iterations_sum += iter_
                sync_iterations_sum += iter_
                disclosed_information_sum += 2 * iter_

            if first_sync_ab is None and iter_ == max_iterations:
                failures += 1

            binary_weights_A = weights_to_binary(TPM_A.weights, Bits)
            binary_weights_B = weights_to_binary(TPM_B.weights, Bits)

            corr_ab = calculate_correlation(TPM_A.weights, TPM_B.weights)
            if not np.isnan(corr_ab):
                correlation_ab_sum[iter_ - 1] += corr_ab
                correlation_ab_count[iter_ - 1] += 1

            mi_ab = calculate_mutual_information(
                binary_weights_A, binary_weights_B
            )
            if not np.isnan(mi_ab):
                mutual_information_ab_sum[iter_ - 1] += mi_ab
                mutual_information_ab_count[iter_ - 1] += 1

            weights_equal_ae = np.array_equal(
                TPM_A.weights.flatten(), TPM_E.weights.flatten()
            )
            if first_sync_eve[eve_attack] is None and weights_equal_ae:
                first_sync_eve[eve_attack] = iter_
                eve_stats[eve_attack]["first_sync_count"] += 1
                eve_stats[eve_attack]["first_sync_iteration_sum"] += iter_

            corr_ae = calculate_correlation(TPM_A.weights, TPM_E.weights)
            if not np.isnan(corr_ae):
                eve_stats[eve_attack]["correlation_sum"][iter_ - 1] += corr_ae
                eve_stats[eve_attack]["correlation_count"][iter_ - 1] += 1

            binary_weights_E = weights_to_binary(TPM_E.weights, Bits)
            mi_ae = calculate_mutual_information(
                binary_weights_A, binary_weights_E
            )
            if not np.isnan(mi_ae):
                eve_stats[eve_attack]["mutual_information_sum"][iter_ - 1] += mi_ae
                eve_stats[eve_attack]["mutual_information_count"][iter_ - 1] += 1

        eve_first_sync = first_sync_eve[eve_attack]
        if (
            first_sync_ab is not None
            and eve_first_sync is not None
            and eve_first_sync < first_sync_ab
        ):
            eve_stats[eve_attack]["faster_than_bob_count"] += 1

    return {
        "failures": failures,
        "sync_iterations_sum": sync_iterations_sum,
        "disclosed_information_sum": disclosed_information_sum,
        "fact_sync_iterations_sum": fact_sync_iterations_sum,
        "correlation_ab_sum": correlation_ab_sum,
        "correlation_ab_count": correlation_ab_count,
        "mutual_information_ab_sum": mutual_information_ab_sum,
        "mutual_information_ab_count": mutual_information_ab_count,
        "eve_stats": eve_stats,
        "leaked_bits_count": leaked_bits_count,
        "measurements": measurements_chunk,
    }


def measure_single_parameter_set_parallel(K, N, Bits, M=None, kStart=1, kLimit=5,
                                           TCounterLimit=50, measurements=100,
                                           max_iterations=1000, Q=0.05,
                                           n_workers=None, verbose=True,
                                           eve_attack="naive",
                                           leakage_enabled=False):
    if eve_attack not in ("naive", "geometric", "both"):
        raise ValueError(
            'eve_attack должен иметь значение "naive", "geometric" или "both"'
        )
    if measurements < 1:
        raise ValueError("measurements должен быть не меньше 1")
    if not 0.0 <= Q <= 1.0:
        raise ValueError("Q должен находиться в диапазоне от 0 до 1")

    L_local = (2 ** Bits) // 2
    if M is None:
        M = L_local
    if not isinstance(M, int) or not 1 <= M <= L_local:
        raise ValueError(f"M должен находиться в диапазоне от 1 до L={L_local}")
    if max_iterations < 1:
        raise ValueError("max_iterations должен быть не меньше 1")

    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, measurements))

    base = measurements // n_workers
    remainder = measurements % n_workers
    chunks = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    chunks = [c for c in chunks if c > 0]

    seeds = [int.from_bytes(os.urandom(4), "little") for _ in chunks]

    tasks = [
        (K, N, Bits, M, chunk, max_iterations, Q, seed, eve_attack,
         leakage_enabled)
        for chunk, seed in zip(chunks, seeds)
    ]

    total_failures = 0
    total_sync_iterations = 0
    total_disclosed_information = 0.0
    total_fact_sync_iterations = 0
    total_measurements = 0
    total_leaked_bits_count = 0

    total_correlation_ab_sum = np.zeros(max_iterations, dtype=np.float64)
    total_correlation_ab_count = np.zeros(max_iterations, dtype=np.int64)
    total_mutual_information_ab_sum = np.zeros(max_iterations, dtype=np.float64)
    total_mutual_information_ab_count = np.zeros(max_iterations, dtype=np.int64)

    active_attacks = _active_attack_names(eve_attack)

    total_eve_stats = {
        attack: {
            "correlation_sum": np.zeros(max_iterations, dtype=np.float64),
            "correlation_count": np.zeros(max_iterations, dtype=np.int64),
            "mutual_information_sum": np.zeros(max_iterations, dtype=np.float64),
            "mutual_information_count": np.zeros(max_iterations, dtype=np.int64),
            "first_sync_count": 0,
            "first_sync_iteration_sum": 0,
            "faster_than_bob_count": 0,
        }
        for attack in active_attacks
    }

    with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
        futures = [ex.submit(_measure_chunk, task) for task in tasks]

        for future in as_completed(futures):
            chunk_result = future.result()

            total_failures += chunk_result["failures"]
            total_sync_iterations += chunk_result["sync_iterations_sum"]
            total_disclosed_information += chunk_result["disclosed_information_sum"]
            total_fact_sync_iterations += chunk_result["fact_sync_iterations_sum"]
            total_measurements += chunk_result["measurements"]
            total_leaked_bits_count += chunk_result["leaked_bits_count"]

            total_correlation_ab_sum += chunk_result["correlation_ab_sum"]
            total_correlation_ab_count += chunk_result["correlation_ab_count"]
            total_mutual_information_ab_sum += chunk_result["mutual_information_ab_sum"]
            total_mutual_information_ab_count += chunk_result["mutual_information_ab_count"]

            for attack in active_attacks:
                chunk_eve = chunk_result["eve_stats"][attack]
                total_eve_stats[attack]["correlation_sum"] += chunk_eve["correlation_sum"]
                total_eve_stats[attack]["correlation_count"] += chunk_eve["correlation_count"]
                total_eve_stats[attack]["mutual_information_sum"] += chunk_eve["mutual_information_sum"]
                total_eve_stats[attack]["mutual_information_count"] += chunk_eve["mutual_information_count"]
                total_eve_stats[attack]["first_sync_count"] += chunk_eve["first_sync_count"]
                total_eve_stats[attack]["first_sync_iteration_sum"] += chunk_eve["first_sync_iteration_sum"]
                total_eve_stats[attack]["faster_than_bob_count"] += chunk_eve["faster_than_bob_count"]

    fer_average = total_failures / total_measurements
    denominator = total_measurements - total_failures

    if denominator > 0:
        fact_sync_iterations_average = total_fact_sync_iterations / denominator
        sync_iterations_average = total_sync_iterations / denominator
        disclosed_information_average = total_disclosed_information / denominator
    else:
        fact_sync_iterations_average = 0
        sync_iterations_average = 0
        disclosed_information_average = 0

    correlation_ab_average = np.full(max_iterations, np.nan)
    np.divide(
        total_correlation_ab_sum,
        total_correlation_ab_count,
        out=correlation_ab_average,
        where=total_correlation_ab_count > 0,
    )

    mutual_information_ab_average = np.full(max_iterations, np.nan)
    np.divide(
        total_mutual_information_ab_sum,
        total_mutual_information_ab_count,
        out=mutual_information_ab_average,
        where=total_mutual_information_ab_count > 0,
    )

    eve_average_metrics = {}
    for attack in active_attacks:
        stats = total_eve_stats[attack]

        correlation_average = np.full(max_iterations, np.nan)
        np.divide(
            stats["correlation_sum"],
            stats["correlation_count"],
            out=correlation_average,
            where=stats["correlation_count"] > 0,
        )

        mutual_information_average = np.full(max_iterations, np.nan)
        np.divide(
            stats["mutual_information_sum"],
            stats["mutual_information_count"],
            out=mutual_information_average,
            where=stats["mutual_information_count"] > 0,
        )

        first_sync_count = stats["first_sync_count"]
        if first_sync_count:
            first_sync_average = (
                stats["first_sync_iteration_sum"] / first_sync_count
            )
        else:
            first_sync_average = 0.0

        faster_count = stats["faster_than_bob_count"]
        faster_percentage = faster_count / total_measurements * 100

        eve_average_metrics[attack] = {
            "correlation": correlation_average,
            "mutual_information": mutual_information_average,
            "first_sync_count": first_sync_count,
            "first_sync_iteration_average": first_sync_average,
            "faster_than_bob_count": faster_count,
            "faster_than_bob_percentage": faster_percentage,
        }

    correlation_data = {
        "iteration": np.arange(1, max_iterations + 1),
        "correlation_Alice_Bob": correlation_ab_average,
    }

    mutual_information_data = {
        "iteration": np.arange(1, max_iterations + 1),
        "mutual_information_Alice_Bob": mutual_information_ab_average,
    }

    for attack in active_attacks:
        suffix = _attack_suffix(attack)
        correlation_data[f"correlation_Alice_Eve_{suffix}"] = (
            eve_average_metrics[attack]["correlation"]
        )
        mutual_information_data[f"mutual_information_Alice_Eve_{suffix}"] = (
            eve_average_metrics[attack]["mutual_information"]
        )

    correlation_data["correlation_Alice_Eve"] = (
        eve_average_metrics[eve_attack]["correlation"]
    )
    mutual_information_data["mutual_information_Alice_Eve"] = (
        eve_average_metrics[eve_attack]["mutual_information"]
    )


    mutual_information_df = pd.DataFrame(mutual_information_data)

    eve_metrics = eve_average_metrics[eve_attack]
    summary = {
        "K": K,
        "N": N,
        "Bits": Bits,
        "M": M,
        "L": L_local,
        "measurements": total_measurements,
        "max_iterations": max_iterations,
        "QBER": Q,
        "eve_attack": eve_attack,
        "leakage_enabled": leakage_enabled,
        "leaked_bits_total": total_leaked_bits_count,
        "leaked_bits_average_per_replication": (
            total_leaked_bits_count / total_measurements
        ),
        "FER_Alice_Bob": fer_average,
        "average_first_sync_iteration_Alice_Bob": fact_sync_iterations_average,
        "eve_first_sync_count": eve_metrics["first_sync_count"],
        "eve_average_first_sync_iteration": eve_metrics[
            "first_sync_iteration_average"
        ],
        "eve_faster_than_bob_count": eve_metrics["faster_than_bob_count"],
        "eve_faster_than_bob_percentage": eve_metrics[
            "faster_than_bob_percentage"
        ],
    }

    summary_df = pd.DataFrame([summary])

    leakage_tag = "on" if leakage_enabled else "off"
    output_file = (
        f"resultK={K},N={N},bits={Bits},M={M},attack={eve_attack},"
        f"leakage={leakage_tag},security.xlsx"
    )
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    if verbose:
        print(
            K, N, Bits,
            "Attack =", eve_attack,
            "FER_AB =", fer_average,
            "sync_AB =", fact_sync_iterations_average,
        )
        for attack in active_attacks:
            metrics = eve_average_metrics[attack]
            print(
                f"Eve_{attack}_faster_count =", metrics["faster_than_bob_count"],
                f"Eve_{attack}_faster_% =", metrics["faster_than_bob_percentage"]
            )
        print("Учет утечки информации:", "включен" if leakage_enabled else "выключен")
        print("Всего раскрыто битов:", total_leaked_bits_count)
        print("Среднее число раскрытых битов на репликацию:", total_leaked_bits_count / total_measurements)
        print("Excel =", output_file)

    result = {
        "K": K,
        "N": N,
        "Bits": Bits,
        "M": M,
        "eve_attack": eve_attack,
        "leakage_enabled": leakage_enabled,
        "leaked_bits_total": total_leaked_bits_count,
        "leaked_bits_average_per_replication": (
            total_leaked_bits_count / total_measurements
        ),
        "fer_average": fer_average,
        "fact_sync_iterations_average": fact_sync_iterations_average,
        "sync_iterations_average": sync_iterations_average,
        "disclosed_information_average": disclosed_information_average,
        "correlation_Alice_Bob": correlation_ab_average,
        "mutual_information_Alice_Bob": mutual_information_ab_average,
        "n_workers": len(chunks),
        "output_file": output_file,
    }

    eve_metrics = eve_average_metrics[eve_attack]
    result["correlation_Alice_Eve"] = eve_metrics["correlation"]
    result["mutual_information_Alice_Eve"] = eve_metrics["mutual_information"]
    result["eve_first_sync_count"] = eve_metrics["first_sync_count"]
    result["eve_average_first_sync_iteration"] = eve_metrics[
        "first_sync_iteration_average"
    ]
    result["eve_faster_count"] = eve_metrics["faster_than_bob_count"]
    result["eve_faster_percentage"] = eve_metrics["faster_than_bob_percentage"]

    return result


def create_key_with_share(source_key, share):
    source_key = np.asarray(source_key, dtype=int)
    n = len(source_key)
    share_count = int(round(n * share))

    result = rnd.randint(0, 2, size=n)

    if share_count > 0:
        shared_positions = rnd.choice(
            n,
            size=share_count,
            replace=False
        )
        result[shared_positions] = source_key[shared_positions]

    return result


def mutual_information_discrete(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    if len(x) != len(y) or len(x) == 0:
        return 0.0

    _, x_inverse, x_counts = np.unique(
        x, return_inverse=True, return_counts=True
    )
    _, y_inverse, y_counts = np.unique(
        y, return_inverse=True, return_counts=True
    )

    pair_index = x_inverse * len(y_counts) + y_inverse
    pair_counts = np.bincount(pair_index)
    pair_ids = np.flatnonzero(pair_counts)

    n = len(x)
    px = x_counts[pair_ids // len(y_counts)] / n
    py = y_counts[pair_ids % len(y_counts)] / n
    pxy = pair_counts[pair_ids] / n

    return float(np.sum(pxy * np.log2(pxy / (px * py))))


def _robustness_pair_metrics(TPM_A, TPM_B, TPM_E):
    return {
        "mutual_information_Alice_Bob": mutual_information_discrete(
            TPM_A.weights.flatten(), TPM_B.weights.flatten()
        ),
        "mutual_information_Alice_Eve": mutual_information_discrete(
            TPM_A.weights.flatten(), TPM_E.weights.flatten()
        ),
        "correlation_Alice_Bob": calculate_correlation(
            TPM_A.weights, TPM_B.weights
        ),
        "correlation_Alice_Eve": calculate_correlation(
            TPM_A.weights, TPM_E.weights
        ),
    }


def _robustness_winner(bob_sync_iteration, eve_sync_iteration):
    if bob_sync_iteration is None and eve_sync_iteration is None:
        return None
    if bob_sync_iteration is None:
        return "eve"
    if eve_sync_iteration is None:
        return "bob"
    if bob_sync_iteration < eve_sync_iteration:
        return "bob"
    if eve_sync_iteration < bob_sync_iteration:
        return "eve"
    return "tie"


def _robustness_measure_chunk(args):
    (
        K, N, Bits, scenario, shared_fraction, M,
        measurements_chunk, max_iterations, seed, trial_start, eve_attack,
        leakage_enabled
    ) = args

    rnd.seed(seed)

    n_local = K * N * Bits
    L_local = (2 ** Bits) // 2
    results = []

    for local_trial in range(measurements_chunk):
        trial_number = trial_start + local_trial + 1

        if scenario == 1:
            keySlice_A = rnd.randint(0, 2, size=n_local)
            keySlice_B = rnd.randint(0, 2, size=n_local)
            keySlice_E = create_key_with_share(keySlice_A, shared_fraction)
        elif scenario == 2:
            keySlice_A = rnd.randint(0, 2, size=n_local)
            keySlice_B = create_key_with_share(keySlice_A, shared_fraction)
            keySlice_E = rnd.randint(0, 2, size=n_local)
        else:
            raise ValueError("scenario должен быть равен 1 или 2")

        eve_key_estimate = np.array(keySlice_E, copy=True)
        weights_A = binary_to_decimal(keySlice_A, Bits)
        weights_B = binary_to_decimal(keySlice_B, Bits)
        weights_E = binary_to_decimal(keySlice_E, Bits)

        arrayWeights_A = list_to_matrix(weights_A, K, N)
        arrayWeights_B = list_to_matrix(weights_B, K, N)
        arrayWeights_E = list_to_matrix(weights_E, K, N)

        TPM_A = TreeParityMachine(K, N, L_local, arrayWeights_A)
        TPM_B = TreeParityMachine(K, N, L_local, arrayWeights_B)
        TPM_E = TreeParityMachine(
            K, N, L_local, np.array(arrayWeights_E, copy=True)
        )

        bob_sync_iteration = None
        if np.array_equal(TPM_A.weights, TPM_B.weights):
            bob_sync_iteration = 0

        eve_sync_iteration = None
        if np.array_equal(TPM_A.weights, TPM_E.weights):
            eve_sync_iteration = 0

        winner = _robustness_winner(bob_sync_iteration, eve_sync_iteration)
        decision_iteration = 0 if winner is not None else None
        metric_snapshot = None
        if winner is not None:
            metric_snapshot = _robustness_pair_metrics(TPM_A, TPM_B, TPM_E)

        iterations = 0
        leaked_bits_count = 0

        while iterations < max_iterations and winner is None:
            iterations += 1

            if Bits > 1:
                vals = np.r_[np.arange(-M, 0), np.arange(1, M + 1)]
                X = np.random.choice(vals, size=(K, N))
            else:
                X = np.random.randint(-1, 2, [K, N])

            out_A = TPM_A.compute_output(X)
            out_B = TPM_B.compute_output(X)
            out_E = TPM_E.compute_output(X)

            if out_A == out_B:
                TPM_A.update(X, out_B)
                TPM_B.update(X, out_A)

            if leakage_enabled and disclose_one_bit_to_eve(
                keySlice_A, eve_key_estimate, [TPM_E], Bits
            ):
                leaked_bits_count += 1
                out_E = TPM_E.compute_output(X)

            _update_eve(TPM_E, X, out_A, out_B, out_E, eve_attack)

            if bob_sync_iteration is None and np.array_equal(
                TPM_A.weights.flatten(), TPM_B.weights.flatten()
            ):
                bob_sync_iteration = iterations

            if eve_sync_iteration is None and np.array_equal(
                TPM_A.weights.flatten(), TPM_E.weights.flatten()
            ):
                eve_sync_iteration = iterations

            winner = _robustness_winner(
                bob_sync_iteration, eve_sync_iteration
            )
            if winner is not None:
                decision_iteration = iterations
                metric_snapshot = _robustness_pair_metrics(
                    TPM_A, TPM_B, TPM_E
                )

        if metric_snapshot is None:
            metric_snapshot = _robustness_pair_metrics(
                TPM_A, TPM_B, TPM_E
            )
            decision_iteration = iterations

        row = {
            "M": M,
            "scenario": scenario,
            "shared_fraction": shared_fraction,
            "shared_fraction_percent": shared_fraction * 100,
            "trial": trial_number,
            "iterations": iterations,
            "leaked_bits": leaked_bits_count,
            "bob_sync_iteration": bob_sync_iteration,
            "alice_bob_sync": int(bob_sync_iteration is not None),
            "eve_attack": eve_attack,
            "winner": winner or "none",
            "decision_iteration": decision_iteration,
            "eve_sync_iteration": eve_sync_iteration,
            "alice_eve_sync": int(eve_sync_iteration is not None),
            "eve_faster": int(winner == "eve"),
            "bob_faster": int(winner == "bob"),
            "tie": int(winner == "tie"),
            "mutual_information_Alice_Bob": metric_snapshot[
                "mutual_information_Alice_Bob"
            ],
            "mutual_information_Alice_Eve": metric_snapshot[
                "mutual_information_Alice_Eve"
            ],
            "correlation_Alice_Bob": metric_snapshot[
                "correlation_Alice_Bob"
            ],
            "correlation_Alice_Eve": metric_snapshot[
                "correlation_Alice_Eve"
            ],
        }
        results.append(row)

    return results


def _run_robustness_round(
        K, N, Bits, scenario, shared_fraction, M, eve_attack,
        measurements=1000, max_iterations=10000, n_workers=None,
        leakage_enabled=False):
    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, measurements))

    base = measurements // n_workers
    remainder = measurements % n_workers
    chunks = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    chunks = [chunk for chunk in chunks if chunk > 0]

    seeds = [int.from_bytes(os.urandom(4), "little") for _ in chunks]
    tasks = []
    trial_start = 0

    for chunk, seed in zip(chunks, seeds):
        tasks.append((
            K, N, Bits, scenario, shared_fraction, M,
            chunk, max_iterations, seed, trial_start, eve_attack,
            leakage_enabled
        ))
        trial_start += chunk

    all_results = []
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(_robustness_measure_chunk, task) for task in tasks]
        for future in as_completed(futures):
            all_results.extend(future.result())

    all_results.sort(key=lambda row: row["trial"])
    return all_results


def _summarize_robustness_round(results, K, N, Bits, M, scenario,
                                round_number, shared_fraction, eve_attack):
    total = len(results)
    bob_sync_iterations = [
        row["bob_sync_iteration"] for row in results
        if row["bob_sync_iteration"] is not None
    ]
    eve_sync_iterations = [
        row["eve_sync_iteration"] for row in results
        if row["eve_sync_iteration"] is not None
    ]
    eve_faster_rows = [row for row in results if row["winner"] == "eve"]
    bob_faster_rows = [row for row in results if row["winner"] == "bob"]
    tie_rows = [row for row in results if row["winner"] == "tie"]

    eve_faster_iterations = [row["eve_sync_iteration"] for row in eve_faster_rows]
    bob_faster_iterations = [row["bob_sync_iteration"] for row in bob_faster_rows]
    mi_ab_values = [row["mutual_information_Alice_Bob"] for row in results]
    mi_ae_values = [row["mutual_information_Alice_Eve"] for row in results]
    corr_ab_values = [
        row["correlation_Alice_Bob"] for row in results
        if np.isfinite(row["correlation_Alice_Bob"])
    ]
    corr_ae_values = [
        row["correlation_Alice_Eve"] for row in results
        if np.isfinite(row["correlation_Alice_Eve"])
    ]

    return {
        "K": K,
        "N": N,
        "Bits": Bits,
        "M": M,
        "scenario": scenario,
        "round": round_number,
        "shared_fraction": shared_fraction,
        "shared_fraction_percent": shared_fraction * 100,
        "eve_attack": eve_attack,
        "experiments": total,
        "leakage_enabled": any(row["leaked_bits"] > 0 for row in results),
        "leaked_bits_total": sum(row["leaked_bits"] for row in results),
        "leaked_bits_average_per_replication": float(
            np.mean([row["leaked_bits"] for row in results])
        ),
        "Alice_Bob_sync_count": len(bob_sync_iterations),
        "Alice_Bob_sync_share": len(bob_sync_iterations) / total,
        "Alice_Bob_sync_iteration_average": (
            float(np.mean(bob_sync_iterations)) if bob_sync_iterations else 0.0
        ),
        "Alice_Eve_sync_count": len(eve_sync_iterations),
        "Alice_Eve_sync_share": len(eve_sync_iterations) / total,
        "Eve_faster_count": len(eve_faster_rows),
        "Eve_faster_share": len(eve_faster_rows) / total,
        "Eve_faster_iteration_average": (
            float(np.mean(eve_faster_iterations)) if eve_faster_iterations else 0.0
        ),
        "Bob_faster_than_Eve_count": len(bob_faster_rows),
        "Bob_faster_than_Eve_share": len(bob_faster_rows) / total,
        "Bob_faster_iteration_average": (
            float(np.mean(bob_faster_iterations)) if bob_faster_iterations else 0.0
        ),
        "Tie_count": len(tie_rows),
        "Tie_share": len(tie_rows) / total,
        "Alice_Eve_sync_iteration_average": (
            float(np.mean(eve_sync_iterations)) if eve_sync_iterations else 0.0
        ),
        "mutual_information_Alice_Bob_average": float(np.mean(mi_ab_values)),
        "mutual_information_Alice_Eve_average": float(np.mean(mi_ae_values)),
        "correlation_Alice_Bob_average": (
            float(np.mean(corr_ab_values)) if corr_ab_values else np.nan
        ),
        "correlation_Alice_Eve_average": (
            float(np.mean(corr_ae_values)) if corr_ae_values else np.nan
        ),
    }


def measure_robustness_experiment_parallel(
        K, N, Bits, M=None, eve_attack="naive", measurements=1000,
        max_iterations=10000, n_workers=None, verbose=True,
        leakage_enabled=False):
    if eve_attack not in ("naive", "geometric", "both"):
        raise ValueError(
            'eve_attack должен иметь значение "naive", "geometric" или "both"'
        )
    if K < 1 or N < 1 or Bits < 1:
        raise ValueError("K, N и Bits должны быть положительными")
    if measurements < 1:
        raise ValueError("measurements должен быть не меньше 1")
    if max_iterations < 1:
        raise ValueError("max_iterations должен быть не меньше 1")

    L_local = (2 ** Bits) // 2
    if M is None:
        M = L_local
    if not isinstance(M, int) or not 1 <= M <= L_local:
        raise ValueError(f"M должен находиться в диапазоне от 1 до L={L_local}")
    n_workers = n_workers or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, measurements))

    scenario_1_fractions = [round(0.90 - 0.09 * i, 2) for i in range(11)]
    scenario_2_fractions = [round(0.10 + 0.16 * i, 2) for i in range(6)]
    rounds = [(1, scenario_1_fractions), (2, scenario_2_fractions)]

    leakage_tag = "on" if leakage_enabled else "off"
    filename = (
        f"robustness_K={K},N={N},bits={Bits},Mmax={M},attack={eve_attack},"
        f"leakage={leakage_tag}.xlsx"
    )
    all_summary_results = []

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for current_M in range(1, M + 1):
            m_summary_results = []

            for scenario, fractions in rounds:
                for round_number, shared_fraction in enumerate(fractions, start=1):
                    results = _run_robustness_round(
                        K=K,
                        N=N,
                        Bits=Bits,
                        scenario=scenario,
                        shared_fraction=shared_fraction,
                        M=current_M,
                        eve_attack=eve_attack,
                        measurements=measurements,
                        max_iterations=max_iterations,
                        n_workers=n_workers,
                        leakage_enabled=leakage_enabled,
                    )

                    summary_row = _summarize_robustness_round(
                        results, K, N, Bits, current_M, scenario,
                        round_number, shared_fraction, eve_attack
                    )
                    summary_row["leakage_enabled"] = leakage_enabled
                    m_summary_results.append(summary_row)
                    all_summary_results.append(summary_row)

                    if verbose:
                        print(
                            f"K={K}, N={N}, Bits={Bits}, M={current_M}, "
                            f"scenario={scenario}, "
                            f"share={shared_fraction:.0%}, "
                            f"experiments={len(results)}"
                        )

            summary_df = pd.DataFrame(m_summary_results)
            sheet_name = f"M_{current_M}"
            summary_df.to_excel(writer, sheet_name=sheet_name, index=False)

    summary_all_df = pd.DataFrame(all_summary_results)

    if verbose:
        print(f"Результаты сохранены: {os.path.abspath(filename)}")

    return {
        "K": K,
        "N": N,
        "Bits": Bits,
        "eve_attack": eve_attack,
        "leakage_enabled": leakage_enabled,
        "measurements_per_point": measurements,
        "max_iterations": max_iterations,
        "M_values": M,
        "M_max": M,
        "L": L_local,
        "summary": summary_all_df,
        "n_workers": n_workers,
        "output_file": filename,
    }


def read_positive_int(prompt, default=None):
    while True:
        value = input(prompt).strip()
        if value == "" and default is not None:
            return default
        try:
            number = int(value)
            if number > 0:
                return number
            print("Ошибка: необходимо ввести положительное целое число.")
        except ValueError:
            print("Ошибка: необходимо ввести целое число.")


def read_float(prompt, default=None, min_value=0.0, max_value=1.0):
    while True:
        value = input(prompt).strip()
        if value == "" and default is not None:
            return default
        try:
            number = float(value.replace(",", "."))
            if math.isfinite(number) and min_value <= number <= max_value:
                return number
            print(f"Ошибка: значение должно находиться от {min_value} до {max_value}.")
        except ValueError:
            print("Ошибка: необходимо ввести вещественное число.")


def read_bounded_positive_int(prompt, max_value, default):
    while True:
        value = input(prompt).strip()
        if value == "":
            return default
        try:
            number = int(value)
            if 1 <= number <= max_value:
                return number
            print(f"Ошибка: значение должно находиться от 1 до {max_value}.")
        except ValueError:
            print("Ошибка: необходимо ввести целое число.")


def main():
    print("=" * 85)
    print("Программа для исследования робастности и устойчивости конфигураций автоматов Кантера-Кинцеля к атакам")
    print("=" * 85)
    print("\nВыберите режим работы программы:")
    print('1. Тесты на сходимость системы «Алиса-Боб» против Евы')
    print("2. Тесты на робастность")
    print("0. Выход")

    while True:
        mode = input("\nВведите номер режима (0-2): ").strip()
        if mode == "0":
            print("Работа программы завершена.")
            return
        if mode in ("1", "2"):
            mode = int(mode)
            break
        print("Ошибка: выберите 1, 2 или 0.")

    print("\nПараметры нейронной сети")
    print("Длина ключа F = b*N*K, где b — число бит при переводе в веса,")
    print("N — количество входов нейрона скрытого слоя, K — число нейронов.")

    Bits = read_positive_int("Введите b (количество бит): ")
    N = read_positive_int("Введите N (количество входов): ")
    K = read_positive_int("Введите K (количество нейронов): ")

    F = Bits * N * K
    L_local = (2 ** Bits) // 2

    if mode == 1:
        M = read_bounded_positive_int(
            f"Введите M (от 1 до L={L_local}) [{L_local}]: ",
            max_value=L_local,
            default=L_local,
        )
        qber = read_float("Введите QBER [0.05]: ", default=0.05)
        measurements = read_positive_int(
            "Введите количество репликаций [100]: ",
            default=100,
        )
        default_iterations = min(1000, F)
    else:
        M = read_bounded_positive_int(
            f"Введите максимальное M для перебора (от 1 до L={L_local}) "
            f"[{L_local}]: ",
            max_value=L_local,
            default=L_local,
        )
        measurements = read_positive_int(
            "Введите количество репликаций на одну долю совпадений [1000]: ",
            default=1000,
        )
        default_iterations = min(10000, F)
        qber = None

    max_iterations = read_bounded_positive_int(
        f"Введите максимальное количество итераций (не более F={F}) "
        f"[{default_iterations}]: ",
        max_value=F,
        default=default_iterations,
    )

    print("\nДополнительный режим:")
    print('Учет информации, утекающей в процессе синхронизации системы "Алиса-Боб"')
    print("0. Выключен (по умолчанию)")
    print("1. Включен")
    while True:
        leakage_choice = input("Учитывать утечку информации? [0]: ").strip() or "0"
        if leakage_choice in ("0", "1"):
            leakage_enabled = leakage_choice == "1"
            break
        print("Ошибка: выберите 0 или 1.")

    print("\nВыберите вариант моделирования Евы:")
    print("1. Наивная синхронизация")
    print("2. Геометрическая атака")
    attack_choices = {"1": "naive", "2": "geometric"}
    while True:
        choice = input("Введите номер варианта (1-2): ").strip()
        if choice in attack_choices:
            eve_attack = attack_choices[choice]
            break
        print("Ошибка: выберите 1-2.")

    print("\nПараметры запуска:")
    print(f"b={Bits}, N={N}, K={K}, F={F}, L={L_local}, M={M}")
    if mode == 1:
        print(f"QBER={qber}")
    else:
        print(f"Репликаций на одну долю совпадений={measurements}")
        print(f"Для тестов на робастность будут проверены M от 1 до {M}")
    print(f"Количество репликаций={measurements}, максимум итераций={max_iterations}")
    print(f"Атака Евы={eve_attack}, учет утечки={'включен' if leakage_enabled else 'выключен'}")

    if mode == 1:
        result = measure_single_parameter_set_parallel(
            K=K,
            N=N,
            Bits=Bits,
            M=M,
            measurements=measurements,
            max_iterations=max_iterations,
            Q=qber,
            eve_attack=eve_attack,
            leakage_enabled=leakage_enabled,
        )
    else:
        print("\nТесты на робастность:")
        print("Сценарий 1: Ева получает заданную долю битов Алисы.")
        print("Сценарий 2: Алиса и Боб имеют заданную долю общих битов.")
        result = measure_robustness_experiment_parallel(
            K=K,
            N=N,
            Bits=Bits,
            M=M,
            eve_attack=eve_attack,
            measurements=measurements,
            max_iterations=max_iterations,
            leakage_enabled=leakage_enabled,
        )

    print("\nЭксперимент завершён.")
    print(f"Результаты сохранены: {os.path.abspath(result['output_file'])}")


if __name__ == "__main__":
    main()