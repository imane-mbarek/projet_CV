import time
import cv2
import numpy as np
from collections import deque


# ─────────────────────────────────────────────
class PersonState:
    """Garde en mémoire l'historique de mouvement d'une seule personne."""

    def __init__(self, person_id: int):
        self.person_id = person_id
        self.positions = deque(maxlen=10)
        self.danger_since: float | None = None


# ─────────────────────────────────────────────
class BehaviorClassifier:
    """Analyse le comportement et déclenche une alerte si nécessaire."""

    def __init__(
        self,
        seuil_horizontal: int = 10,
        seuil_vertical:   int = 15,
        duree_alerte:     float = 5.0
    ):
        self.seuil_horizontal = seuil_horizontal
        self.seuil_vertical   = seuil_vertical
        self.duree_alerte     = duree_alerte
        self.etats: dict[int, PersonState] = {}

    def update(self, person_id: int, cx: int, cy: int) -> bool:
        """Met à jour la position d'une personne. Retourne True si danger."""
        if person_id not in self.etats:
            self.etats[person_id] = PersonState(person_id)

        etat = self.etats[person_id]
        etat.positions.append((cx, cy))

        if len(etat.positions) < 10:
            return False

        return self._analyser(etat)

    def _analyser(self, etat: PersonState) -> bool:
        """Vérifie si le mouvement correspond à une noyade."""
        positions = list(etat.positions)

        ancienne_pos = positions[0]
        nouvelle_pos = positions[-1]

        mvt_horizontal = abs(nouvelle_pos[0] - ancienne_pos[0])
        mvt_vertical   = np.std([p[1] for p in positions])

        en_danger = (
            mvt_horizontal < self.seuil_horizontal and
            mvt_vertical   > self.seuil_vertical
        )

        now = time.time()
        if en_danger:
            if etat.danger_since is None:
                etat.danger_since = now
            return (now - etat.danger_since) >= self.duree_alerte
        else:
            etat.danger_since = None
            return False


# ─────────────────────────────────────────────
def draw_alerte(frame: np.ndarray, person_id: int) -> None:
    """Affiche un message d'alerte rouge sur l'écran."""
    hauteur, largeur = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (largeur, 60), (0, 0, 200), -1)

    cv2.putText(
        frame,
        f"ALERTE NOYADE — ID {person_id} !",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, (255, 255, 255), 3, cv2.LINE_AA
    )