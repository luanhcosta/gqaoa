import pennylane as qml
from pennylane import qaoa as qaoa_pennylane
from pennylane import numpy as pnp


class QAOA:
    def __init__(
                 self,
                 expected_value,
                 cov_matrix,
                 q,
                 B,
                 lamb,
                 initial_state='uniform_superposition',
                 mixture_layer='x',
                 edges_hc=None,
                 edges_hb=None
                ):

        self.count_qpu_call = 0

        self.q = q
        self.B = B
        self.lamb = lamb
        self.expected_value = expected_value
        self.cov_matrix = cov_matrix
        self.n_assets = len(expected_value)
        self.initial_state = initial_state
        self.mixture_layer = mixture_layer

        self.edges_complete = [(i, j) for i in range(self.n_assets) for j in range(i+1, self.n_assets)]
        if edges_hc is None:
            self.edges_hc = self.edges_complete
        else:
            self.edges_hc = edges_hc
        if edges_hb is None:
            self.edges_hb = self.edges_complete
        else:
            self.edges_hb = edges_hb

        H_C0 = H_C1 = 0
        for e in self.edges_hc:  H_C0 = H_C0 + self.cost_hamiltonian_wheight(e[0], e[1])*qml.PauliZ(e[0]) @ qml.PauliZ(e[1])
        # range(4), not range(self.n_assets): kept verbatim from the original —
        # changing it would change energy_min for every existing experiment result.
        for q in range(4): H_C1 = H_C1 + self.cost_hamiltonian_wheight(q)*qml.PauliZ(q)
        self.H_C = H_C0+H_C1

    def cost_hamiltonian_wheight(self, i, j=None):
        """
        Calculate the weights for the Hamiltonian of the QUBO problem.

        Args:
            i (int): Index of the first asset.
            j (int, optional): Index of the second asset. Defaults to None.

        Returns:
            float: The weight that multiplies the product of the Z operators, acting on qubits i and j.
        """

        if j is None:
            response =self.expected_value[i]+self.lamb*(2*self.B-self.n_assets)-self.q*self.cov_matrix[i].sum()
        else:
            response = self.q*self.cov_matrix[i][j]+self.lamb
        return response

    def _add_mixture_layer(self, beta):

        if self.mixture_layer == 'x':
            for q in range(self.n_assets):
                qml.RX(2 * beta, wires=q)
        elif self.mixture_layer == 'xy':
            for e in self.edges_hb:
                qml.IsingXY(4*beta, wires=[e[0], e[1]])

    def qaoa_layer(self, gamma, beta):
        qaoa_pennylane.cost_layer(gamma, self.H_C)
        self._add_mixture_layer(beta)

    def prepare_uniform_superposition(self):
        for q in range(self.n_assets): qml.Hadamard(wires=q)

    def _SCS(self, m, k):
        """Implements the Split & Cycle shift unitary."""

        # Two-qubit gate (original: CNOT[m-1 -> m], now: CNOT[m-2 -> m-1])
        qml.CNOT(wires=[m - 2, m - 1])
        qml.CRY(2 * pnp.arccos(pnp.sqrt(1 / m)), wires=[m - 1, m - 2])
        qml.CNOT(wires=[m - 2, m - 1])

        # k-1 three-qubit gates
        for l in range(2, k + 1):
            # original: m-l → target, m → control
            # shift down: (m-1) → control, (m-l-1) → target
            qml.CNOT(wires=[m - l - 1, m - 1])

            qml.ctrl(
                qml.RY,
                control=(m - 1, m - l)
            )(
                2 * pnp.arccos(pnp.sqrt(l / m)),
                wires=m - l - 1
            )

            qml.CNOT(wires=[m - l - 1, m - 1])

    def prepare_dicke_state(self, n, B):
        """Prepares a Dicke(n, B) state using zero-based indexing."""

        # Prepare |1> on the last B qubits: wires n-B, ..., n-1
        for wire_idx in range(n - B, n):
            qml.X(wires=wire_idx)

        # First SCS sequence: original i in [B+1, ..., n] → now (i)
        for i in reversed(range(B + 1, n + 1)):
            self._SCS(i, B)

        # Second SCS sequence: original i in [2, ..., B] → now same
        for i in reversed(range(2, B + 1)):
            self._SCS(i, i - 1)

    def circuit(self, params):
        if self.initial_state == 'uniform_superposition':
            self.prepare_uniform_superposition()
        elif self.initial_state == 'dicke_state':
            self.prepare_dicke_state(self.n_assets, self.B)

        if len(params.shape) == 1:
            depth = int(len(params)/2)
            gamma = params[:depth]
            beta = params[depth:]
        else:
            depth = len(params[0])
            gamma = params[0]
            beta = params[1]

        qml.layer(self.qaoa_layer, depth, gamma, beta)

    def probability_circuit(self, params):
        self.count_qpu_call += 1
        self.circuit(params)
        return qml.probs(wires=range(self.n_assets))

    def cost_function(self, params):
        self.count_qpu_call += 1
        self.circuit(params)
        return qml.expval(self.H_C)
