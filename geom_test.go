package geom

import "testing"

func TestVector(t *testing.T) {
	if got, want := len(V()), 3; got != want {
		t.Fatalf("length of a vector is wrong: got=%d, want=%d", got, want)
	}

	v := V(1, 2, 3)
	v2 := V(2, 4, 6)

	if !v.Add(v).Equals(v2) {
		t.Errorf("addition failed: %v+%v != %v", v, v, v2)
	}
	if !v.Add(v).Sub(v2).Equals(ZeroV) {
		t.Errorf("subtraction failed: %v+%v-%v != 0", v, v, v2)
	}
}

func TestMatrix(t *testing.T) {
	if got, want := len(M()), 9; got != want {
		t.Fatalf("length of a matrix is wrong: got=%d, want=%d", got, want)
	}

	v := Y(2)
	u := Z(2)
	rX := RX(Degrees(45))
	if w := rX.XM(rX).XV(v); !w.Equals(u) {
		t.Errorf("rX*rX*[X] != [Y]: %v != %v", w, u)
	}

	r := V(3, 2, 1)
	m := rX.XM(rX).XV(r)
	if v := V(3, -1, 2); !m.Equals(v) {
		t.Errorf("rX*rX*v != v: %v != %v", m, v)
	}

	rY := RY(Degrees(30)).XM(RY(Degrees(60)))
	m = rY.XV(m)
	if v := V(2, -1, -3); !m.Equals(v) {
		t.Errorf("rY*rX*rX*v != v: %v != %v", m, v)
	}

	rZ := RZ(Degrees(-13)).XM(RZ(Degrees(103)))
	m = rZ.XV(m)
	if v := V(1, 2, -3); !m.Equals(v) {
		t.Errorf("rZ*rY*rX*rX*v != v: %v != %v", m, v)
	}
}

func TestInv(t *testing.T) {
	vs := []struct {
		a, b, c float64
	}{
		{10, 0, 30},
		{30, 10, 0},
		{10, 30, 0},
	}
	for i, v := range vs {
		a := RX(Degrees(v.a)).XM(RY(Degrees(v.b))).XM(RZ(Degrees(v.c)))
		b := RZ(Degrees(-v.c)).XM(RY(Degrees(-v.b))).XM(RX(Degrees(-v.a)))
		if !a.XM(b).Equals(I) {
			t.Fatalf("[%d] test case failed to validate as inverted", i)
		}
		if c, err := a.Inv(); err != nil {
			t.Errorf("[%d] test case failed to invert: %v", i, err)
		} else if !a.XM(c).Equals(I) {
			t.Errorf("[%d] test case inverse wrong: got=%v want=%v", i, c, b)
		}
	}
}

func TestDet(t *testing.T) {
	m := RX(Degrees(13)).XM(RY(Degrees(17))).XM(RZ(Degrees(19)))
	if d := m.Det(); !Zeroish(d - 1) {
		t.Errorf("det(%v) = %v != 1", m, d)
	}
}

func TestEigen(t *testing.T) {
	vs := []struct {
		ax, ay, az float64
	}{
		{90, 0, 0},
		{0, 90, 0},
		{0, 0, 90},
		{10, 10, 10},
		{10, 9, 8},
		{1, 2, 4},
		{1, -2, 4},
	}

	for i, v := range vs {
		r := RX(Degrees(v.ax)).XM(RY(Degrees(v.ay))).XM(RZ(Degrees(v.az)))
		s, v, a, err := r.Eigen()
		if err != nil {
			t.Errorf("[%d] failed to generate eigenvector: %v", i, err)
			continue
		}
		if !Zeroish(s - 1) {
			t.Errorf("[%d] determinant is not 1: %g", i, s)
			continue
		}
		if confirm := r.XV(v); !v.Equals(confirm) {
			t.Errorf("[%d] engenvector %v of %v is not", i, v, r)
			continue
		}

		if rC, err := v.RV(a); err != nil {
			t.Errorf("[%d] failed to do rotation ang=%v around %v: %v", i, a, v, err)
		} else if !rC.Equals(r) {
			t.Errorf("[%d] rotation %v does not match intended for ang=%v around %v", i, rC, a, v)
		}
	}
}
