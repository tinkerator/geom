// Package geom implements matrix math for 3D vectors and matrices.
package geom

import (
	"errors"
	"fmt"
	"math"
)

// zeroEnough defines how close to zero is zero enough. It is set via
// DefineZeroish.
var zeroEnough = 1e-7

// Zeroish confirms that two numbers are close enough to zero. The
// tolerance is set via DefineZeroish.
func Zeroish(a float64) bool {
	return math.Abs(a) < zeroEnough
}

// DefineZeroish sets the zeroEnough scale. It should be called with a
// positive value to have any effect. If called with zero or a
// negative number, it prevents the Zeroish() funcion from recognizing
// near zero. Zeroish defaults to treating 10^-7 as close enough to zero.
func DefineZeroish(z float64) {
	zeroEnough = z
}

// RefZero returns the current zero-enough value in use by the geom
// package.
func RefZero() float64 {
	return zeroEnough
}

// Vector is a column vector with 3 elements.
type Vector []float64

// V defines a vector, explicitly specifying the elements in 0 to 2
// order.
func V(v ...float64) Vector {
	u := make(Vector, 3)
	for i := 0; i < len(v) && i < 3; i++ {
		u[i] = v[i]
	}
	return u
}

// ZeroV is a zero vector.
var ZeroV = V()

// AddS adds a scaled (s) vector (u) to v and allocates a new result.
func (v Vector) AddS(u Vector, s float64) Vector {
	return Vector{
		v[0] + u[0]*s,
		v[1] + u[1]*s,
		v[2] + u[2]*s,
	}
}

// Add adds two vectors (u) and (v).
func (v Vector) Add(u Vector) Vector {
	return v.AddS(u, 1)
}

// Sub subtracts vector (u) from (v).
func (v Vector) Sub(u Vector) Vector {
	return v.AddS(u, -1)
}

// Scale maginfies a vector, v, by scale, s.
func (v Vector) Scale(s float64) Vector {
	return Vector{
		s * v[0],
		s * v[1],
		s * v[2],
	}
}

// X returns an X vector of the specified length.
func X(s float64) Vector {
	return V(s)
}

// Y returns an Y vector of the specified length.
func Y(s float64) Vector {
	return V(0, s)
}

// Z returns an Z vector of the specified length.
func Z(s float64) Vector {
	return V(0, 0, s)
}

// Dot returns the scalar dot product of two vectors.
func (v Vector) Dot(u Vector) float64 {
	return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
}

// R returns the scalar length of a vector.
func (v Vector) R() float64 {
	return math.Sqrt(v.Dot(v))
}

// ErrNormalNotPossible etc are errors returned by this package.
var (
	ErrNormalNotPossible = errors.New("normal not possible")
	ErrNotValidMatrix    = errors.New("not a 3x3 matrix")
	ErrSingular          = errors.New("too close to singular")
)

// Normalize attempts to scale v to return a unit vector parallel to
// v.
func (v Vector) Normalize() (u Vector, err error) {
	r := v.R()
	if Zeroish(r) {
		err = ErrNormalNotPossible
		return
	}
	u = v.Scale(1.0 / r)
	return
}

// Cross returns the cross-product of two vectors. It returns a vector
// perpendicular to both u and v. It returns v x u using right hand
// rule.
func (v Vector) Cross(u Vector) Vector {
	return Vector{
		v[1]*u[2] - v[2]*u[1],
		v[2]*u[0] - v[0]*u[2],
		v[0]*u[1] - v[1]*u[0],
	}
}

// ErrNoNormalForZeroVector indicates that we can't make a vector
// perpendicular to zero.
var ErrNoNormalForZeroVector = errors.New("no non-collinear of zero vector")

// NonCollinear returns a vector that is not a scalar multiple of the
// input.
func (v Vector) NonCollinear() (u Vector, err error) {
	if v.Equals(ZeroV) {
		err = ErrNoNormalForZeroVector
		return
	}
	u = Vector{v[1], v[2], v[0]}
	return
}

// Equals confirms two vectors are close enough to equal.
func (v Vector) Equals(u Vector) bool {
	if len(u) != len(v) {
		panic("two vectors are of different lengths")
	}
	for i, x := range v {
		if !Zeroish(x - u[i]) {
			return false
		}
	}
	return true
}

// Matrix is a square 3x3 matrix type.
type Matrix []float64

// M defines a matrix. It takes a variable number of arguments, if the
// number of arguments is:
//
//  1 - create a matrix with this value used by the elements on the
//      trace.
//
//  3 - create a matrix with these values used to fill the trace elements.
//
//  9 - each element is specified, left to right, top to bottom.
//
//  other - start filling at [0][0], walking rows left to right until
//          out of elements.
func M(v ...float64) Matrix {
	switch len(v) {
	case 1:
		return Matrix{
			v[0], 0, 0,
			0, v[0], 0,
			0, 0, v[0],
		}
	case 3:
		return Matrix{
			v[0], 0, 0,
			0, v[1], 0,
			0, 0, v[2],
		}
	default:
		m := make(Matrix, 9)
		for i := 0; i < len(v) && i < 9; i++ {
			m[i] = v[i]
		}
		return m
	}
}

// I is the identity matrix.
var I = M(1)

// ZeroM is the zero matrix.
var ZeroM = M()

// Scale multiplies a matrix by a scalar, s.
func (m Matrix) Scale(s float64) Matrix {
	n := make(Matrix, 9)
	for i, x := range m {
		n[i] = s * x
	}
	return n
}

// AddS adds two matrices scaling the n by s.
func (m Matrix) AddS(n Matrix, s float64) Matrix {
	if len(m) != len(n) {
		panic("two matrices are of different sizes")
	}
	p := make(Matrix, len(m))
	for i, x := range m {
		p[i] = x + s*n[i]
	}
	return p
}

// Add adds two matrices.
func (m Matrix) Add(n Matrix) Matrix {
	return m.AddS(n, 1)
}

// Sub computes the difference of two matrices.
func (m Matrix) Sub(n Matrix) Matrix {
	return m.AddS(n, -1)
}

// Multiply two matrices, allocating a new one.
func (m Matrix) XM(n Matrix) Matrix {
	return Matrix{
		m[0]*n[0] + m[1]*n[3] + m[2]*n[6],
		m[0]*n[1] + m[1]*n[4] + m[2]*n[7],
		m[0]*n[2] + m[1]*n[5] + m[2]*n[8],

		m[3]*n[0] + m[4]*n[3] + m[5]*n[6],
		m[3]*n[1] + m[4]*n[4] + m[5]*n[7],
		m[3]*n[2] + m[4]*n[5] + m[5]*n[8],

		m[6]*n[0] + m[7]*n[3] + m[8]*n[6],
		m[6]*n[1] + m[7]*n[4] + m[8]*n[7],
		m[6]*n[2] + m[7]*n[5] + m[8]*n[8],
	}
}

// Multiply a vector by a matrix.
func (m Matrix) XV(v Vector) Vector {
	return Vector{
		m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
		m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
		m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
	}
}

// Equals confirms two matrices are close enough to equal.
func (m Matrix) Equals(n Matrix) bool {
	if len(m) != len(n) {
		panic("two matrices are of different sizes")
	}
	for i, x := range m {
		if !Zeroish(x - n[i]) {
			return false
		}
	}
	return true
}

// Transpose returns the transpose of a matrix.
func (m Matrix) Transpose() Matrix {
	return Matrix{
		m[0], m[3], m[6],
		m[1], m[4], m[7],
		m[2], m[5], m[8],
	}
}

// Det returns the determinant of a matrix.
func (m Matrix) Det() float64 {
	return m[0]*(m[4]*m[8]-m[5]*m[7]) +
		m[1]*(m[5]*m[6]-m[3]*m[8]) +
		m[2]*(m[3]*m[7]-m[6]*m[4])
}

// Inv returns the inverse of a matrix. We use the Gauss-Jordan
// method to find the inverse.
func (m Matrix) Inv() (Matrix, error) {
	// start by zeroing out the lower left of m.
	a := M(m...)
	b := M(I...)

	swapRow := func(i, j int) {
		from := i * 3
		to := j * 3
		var spare [3]float64
		copy(spare[:3], a[from:from+3])
		copy(a[from:from+3], a[to:to+3])
		copy(a[to:to+3], spare[:3])
		copy(spare[:3], b[from:from+3])
		copy(b[from:from+3], b[to:to+3])
		copy(b[to:to+3], spare[:3])
	}

	scaleRow := func(i int, s float64) {
		from := i * 3
		a[from] *= s
		a[from+1] *= s
		a[from+2] *= s
		b[from] *= s
		b[from+1] *= s
		b[from+2] *= s
	}

	addRow := func(i, j int, s float64) {
		from := i * 3
		to := j * 3
		a[from] += a[to] * s
		a[from+1] += a[to+1] * s
		a[from+2] += a[to+2] * s
		b[from] += b[to] * s
		b[from+1] += b[to+1] * s
		b[from+2] += b[to+2] * s
	}

	// Ensure top left element is non-zero.
	if Zeroish(a[0]) {
		if Zeroish(a[3]) {
			if Zeroish(a[6]) {
				return b, ErrSingular
			}
			swapRow(2, 0)
		} else {
			swapRow(1, 0)
		}
	}
	scaleRow(0, 1.0/a[0])
	addRow(1, 0, -a[3])
	addRow(2, 0, -a[6])

	// Ensure middle element is non-zero.
	if Zeroish(a[4]) {
		if Zeroish(a[7]) {
			return b, ErrSingular
		}
		swapRow(1, 2)
	}
	scaleRow(1, 1.0/a[4])
	addRow(2, 1, -a[7])

	if Zeroish(a[8]) {
		return b, ErrSingular
	}
	scaleRow(2, 1.0/a[8])

	// Clear the upper right triangle.
	addRow(0, 1, -a[1])
	addRow(0, 2, -a[2])
	addRow(1, 2, -a[5])

	return b, nil
}

// Angle represents an angle.
type Angle float64

// Rad returns an angle in radians.
func (a Angle) Rad() float64 {
	return float64(a)
}

// Deg returns an angle in degrees.
func (a Angle) Deg() float64 {
	return float64(a) * 180.0 / math.Pi
}

// LikeTan returns an angle that is 180 degrees opposed to a. If a is
// non-positive, we add 180 deg, otherwise, we subtract 180 deg.
func (a Angle) LikeTan() Angle {
	if a > 0 {
		return a - math.Pi
	}
	return a + math.Pi
}

// LikeCos returns the angle that shares a cosine with a.
func (a Angle) LikeCos() Angle {
	return -a
}

// LikeSin returns the angle that shares a sine with a.
func (a Angle) LikeSin() Angle {
	if a < 0 {
		return -math.Pi - a
	}
	return math.Pi - a
}

// Degrees returns an angle of the specified degrees.
func Degrees(d float64) Angle {
	return Angle(d * math.Pi / 180.0)
}

// Radians returns an angle of the specified degrees.
func Radians(r float64) Angle {
	return Angle(r)
}

// String displays an angle in degrees rounded to 2 decimal places.
func (a Angle) String() string {
	return fmt.Sprintf("%.2f", a.Deg())
}

// C returns the cosine of an angle.
func (a Angle) C() float64 {
	return math.Cos(a.Rad())
}

// S returns the sine of an angle.
func (a Angle) S() float64 {
	return math.Sin(a.Rad())
}

// T returns the tangent of an angle.
func (a Angle) T() float64 {
	return math.Tan(a.Rad())
}

// X returns the X vector column of a matrix.
func (m Matrix) X() Vector {
	return Vector{m[0], m[3], m[6]}
}

// Y returns the Y vector column of a matrix.
func (m Matrix) Y() Vector {
	return Vector{m[1], m[4], m[7]}
}

// Z returns the Z vector column of a matrix.
func (m Matrix) Z() Vector {
	return Vector{m[2], m[5], m[8]}
}

// RX returns a right-hand rotation matrix around the X axis of angle a.
func RX(a Angle) Matrix {
	c, s := a.C(), a.S()
	return Matrix{
		1, 0, 0,
		0, c, -s,
		0, s, c,
	}
}

// RY returns a right-hand rotation matrix around the Y axis of angle a.
func RY(a Angle) Matrix {
	c, s := a.C(), a.S()
	return Matrix{
		c, 0, s,
		0, 1, 0,
		-s, 0, c,
	}
}

// RZ returns a right-hand rotation matrix around the Z axis of angle a.
func RZ(a Angle) Matrix {
	c, s := a.C(), a.S()
	return Matrix{
		c, -s, 0,
		s, c, 0,
		0, 0, 1,
	}
}

// eigenU computes the eigenvector of a simplified set of equations:
//
//  (1) a*x = b*y + c*z   (for a != 0)
//  (2) d*y = e*z + f*x
//  (3)   1 = x*x + y*y + z*z
//
func eigenU(a, b, c, d, e, f float64) Vector {
	// Use (1) to rewrite (2) as an equation for y and z.
	// That is, d*y = e*z + f/a*(b*y + c*z), or
	//  y*(d*a - f*b) = z*(e*a + f*c)
	//            y*p = z*q
	p := d*a - f*b
	q := e*a + f*c
	if !Zeroish(p) {
		// y = z*q/p
		// (3) 1 = ((b/a*q/p + c/a)^2 + (q/p)^2 + 1)*z^2.
		r1 := (b*q/p + c) / a
		r2 := q / p
		v := Z(math.Sqrt(1.0 / (r1*r1 + r2*r2 + 1)))
		v[1] = v[2] * r2
		v[0] = (b*v[1] + c*v[2]) / a
		return v
	} else if !Zeroish(q) {
		// z = y*p/q
		// (3) 1 = ((b/a + c/a*p/q)^2 + 1 + (p/q)^2)*y^2.
		r1 := (b + c*p/q) / a
		r2 := p / q
		v := Y(math.Sqrt(1.0 / (r1*r1 + 1 + r2*r2)))
		v[2] = v[1] * r2
		v[0] = (b*v[1] + c*v[2]) / a
		return v
	}
	return Y(1)
}

// Eigen decomposes a matrix into its determinant (s), its eigenvector
// (v) and an angle (a) which is the positive angle or rotation around
// v that the matrix represents.
func (m Matrix) Eigen() (s float64, v Vector, a Angle, err error) {
	if len(m) != 9 {
		err = ErrNotValidMatrix
		return
	}

	s = m.Det()
	if Zeroish(s) {
		err = ErrSingular
		return
	}

	// normalized matrix.
	n := m.Scale(1 / s)

	if d := 1 - n[0]; !Zeroish(d) {
		// v[0] = (n[1]*v[1] + n[2]*v[2]) / (1-n[0])
		v = eigenU(d, n[1], n[2], 1-n[4], n[5], n[3])
	} else if d = 1 - n[4]; !Zeroish(d) {
		// v[1] = (n[5]*v[2] + n[3]*v[0]) / (1-n[4])
		v = eigenU(d, n[5], n[3], 1-n[8], n[6], n[7])
		v[0], v[1], v[2] = v[2], v[0], v[1]
	} else {
		err = errors.New("too close to identity")
		return
	}

	// v holds the normalized eigenvector. Next we need to figure
	// out what angle around this vector, the matrix rotation
	// represents.
	u, err2 := v.NonCollinear()
	if err2 != nil {
		err = err2
		return
	}
	perp, _ := u.Cross(v).Normalize()
	r := m.XV(perp)
	axis, _ := perp.Cross(r).Normalize()

	a = Radians(math.Acos(perp.Dot(r)))
	v = v.Scale(v.Dot(axis))

	return
}

// RV returns a rotation matrix for an angle (a) around vector axis
// (v). The axis vector (v) need not be of unit length. This function
// uses the Rodrigues rotation formula.
func (v Vector) RV(a Angle) (m Matrix, err error) {
	u, err2 := v.Normalize()
	if err2 != nil {
		err = err2
		return
	}
	w := Matrix{
		0, -u[2], u[1],
		u[2], 0, -u[0],
		-u[1], u[0], 0,
	}

	w2 := w.XM(w)
	m = I.AddS(w, a.S()).AddS(w2, 1-a.C())
	return
}
