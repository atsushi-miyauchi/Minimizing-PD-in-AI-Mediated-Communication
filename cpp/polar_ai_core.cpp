#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace {

class Buffer {
 public:
  Py_buffer view{};
  bool acquired = false;

  ~Buffer() {
    if (acquired) {
      PyBuffer_Release(&view);
    }
  }

  bool read(PyObject* object) {
    acquired = PyObject_GetBuffer(
                   object, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) == 0;
    return acquired;
  }

  bool write(PyObject* object) {
    acquired = PyObject_GetBuffer(
                   object, &view,
                   PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) == 0;
    return acquired;
  }

  Py_ssize_t size() const { return view.len / view.itemsize; }
};

bool require_double(const Buffer& buffer, const char* name) {
  if (buffer.view.itemsize != static_cast<Py_ssize_t>(sizeof(double))) {
    PyErr_Format(PyExc_TypeError, "%s must contain float64 values", name);
    return false;
  }
  return true;
}

bool require_indices(const Buffer& buffer, const char* name) {
  if (buffer.view.itemsize != 4 && buffer.view.itemsize != 8) {
    PyErr_Format(PyExc_TypeError, "%s must contain 32- or 64-bit integers", name);
    return false;
  }
  return true;
}

int64_t index_at(const Buffer& buffer, Py_ssize_t position) {
  if (buffer.view.itemsize == 4) {
    return static_cast<const int32_t*>(buffer.view.buf)[position];
  }
  return static_cast<const int64_t*>(buffer.view.buf)[position];
}

double clipped_product(double alpha, double z) {
  const double value = alpha * z;
  return value < -1.0 ? -1.0 : (value > 1.0 ? 1.0 : value);
}

PyObject* extremize(PyObject*, PyObject* args) {
  PyObject *z_object, *alpha_object, *out_object;
  if (!PyArg_ParseTuple(args, "OOO", &z_object, &alpha_object, &out_object)) {
    return nullptr;
  }
  Buffer z, alpha, out;
  if (!z.read(z_object) || !alpha.read(alpha_object) || !out.write(out_object) ||
      !require_double(z, "z") || !require_double(alpha, "alpha") ||
      !require_double(out, "out")) {
    return nullptr;
  }
  if (z.size() != alpha.size() || z.size() != out.size()) {
    PyErr_SetString(PyExc_ValueError, "z, alpha, and out must have equal length");
    return nullptr;
  }
  const auto* z_data = static_cast<const double*>(z.view.buf);
  const auto* alpha_data = static_cast<const double*>(alpha.view.buf);
  auto* out_data = static_cast<double*>(out.view.buf);
  const Py_ssize_t n = z.size();
  PyThreadState* state = PyEval_SaveThread();
  for (Py_ssize_t i = 0; i < n; ++i) {
    out_data[i] = clipped_product(alpha_data[i], z_data[i]);
  }
  PyEval_RestoreThread(state);
  Py_RETURN_NONE;
}

PyObject* compute_equilibrium(PyObject*, PyObject* args) {
  PyObject *indptr_object, *indices_object, *data_object, *s_object;
  PyObject *lambda_object, *alpha_object, *initial_object, *out_object;
  double tolerance;
  int max_iterations;
  if (!PyArg_ParseTuple(args, "OOOOOOOOdi", &indptr_object, &indices_object,
                        &data_object, &s_object, &lambda_object, &alpha_object,
                        &initial_object, &out_object, &tolerance,
                        &max_iterations)) {
    return nullptr;
  }

  Buffer indptr, indices, data, s, lambda_vec, alpha, initial, out;
  if (!indptr.read(indptr_object) || !indices.read(indices_object) ||
      !data.read(data_object) || !s.read(s_object) ||
      !lambda_vec.read(lambda_object) || !alpha.read(alpha_object) ||
      !initial.read(initial_object) || !out.write(out_object) ||
      !require_indices(indptr, "indptr") ||
      !require_indices(indices, "indices") || !require_double(data, "data") ||
      !require_double(s, "s") || !require_double(lambda_vec, "lambda_vec") ||
      !require_double(alpha, "alpha") ||
      !require_double(initial, "initial") || !require_double(out, "out")) {
    return nullptr;
  }

  const Py_ssize_t n = s.size();
  if (indptr.size() != n + 1 || lambda_vec.size() != n || alpha.size() != n ||
      initial.size() != n || out.size() != n || indices.size() != data.size()) {
    PyErr_SetString(PyExc_ValueError, "incompatible CSR or vector dimensions");
    return nullptr;
  }

  const auto* matrix_data = static_cast<const double*>(data.view.buf);
  const auto* s_data = static_cast<const double*>(s.view.buf);
  const auto* lambda_data = static_cast<const double*>(lambda_vec.view.buf);
  const auto* alpha_data = static_cast<const double*>(alpha.view.buf);
  const auto* initial_data = static_cast<const double*>(initial.view.buf);
  auto* out_data = static_cast<double*>(out.view.buf);

  auto* z = static_cast<double*>(malloc(sizeof(double) * n));
  auto* next = static_cast<double*>(malloc(sizeof(double) * n));
  auto* mediated = static_cast<double*>(malloc(sizeof(double) * n));
  auto* base = static_cast<double*>(malloc(sizeof(double) * n));
  if (z == nullptr || next == nullptr || mediated == nullptr || base == nullptr) {
    free(z);
    free(next);
    free(mediated);
    free(base);
    return PyErr_NoMemory();
  }
  memcpy(z, initial_data, sizeof(double) * n);
  int iterations = 0;
  bool converged = false;
  double residual = HUGE_VAL;

  PyThreadState* state = PyEval_SaveThread();
  for (Py_ssize_t i = 0; i < n; ++i) {
    base[i] = (1.0 - lambda_data[i]) * s_data[i];
  }
  for (iterations = 1; iterations <= max_iterations; ++iterations) {
    for (Py_ssize_t i = 0; i < n; ++i) {
      mediated[i] = clipped_product(alpha_data[i], z[i]);
    }
    residual = 0.0;
    for (Py_ssize_t row = 0; row < n; ++row) {
      double sum = 0.0;
      const auto begin = index_at(indptr, row);
      const auto end = index_at(indptr, row + 1);
      for (int64_t p = begin; p < end; ++p) {
        const auto column = index_at(indices, p);
        sum += matrix_data[p] * mediated[column];
      }
      next[row] = base[row] + lambda_data[row] * sum;
      const double difference = fabs(next[row] - z[row]);
      residual = difference > residual ? difference : residual;
    }
    double* swap = z;
    z = next;
    next = swap;
    if (residual < tolerance) {
      converged = true;
      break;
    }
  }
  if (iterations > max_iterations) {
    iterations = max_iterations;
  }
  memcpy(out_data, z, sizeof(double) * n);
  PyEval_RestoreThread(state);
  free(z);
  free(next);
  free(mediated);
  free(base);

  return Py_BuildValue("(iOd)", iterations, converged ? Py_True : Py_False,
                       residual);
}

PyObject* polarization(PyObject*, PyObject* args) {
  PyObject* z_object;
  if (!PyArg_ParseTuple(args, "O", &z_object)) {
    return nullptr;
  }
  Buffer z;
  if (!z.read(z_object) || !require_double(z, "z")) {
    return nullptr;
  }
  const auto* z_data = static_cast<const double*>(z.view.buf);
  double value = 0.0;
  const Py_ssize_t n = z.size();
  PyThreadState* state = PyEval_SaveThread();
  for (Py_ssize_t i = 0; i < n; ++i) {
    value += z_data[i] * z_data[i];
  }
  PyEval_RestoreThread(state);
  return PyFloat_FromDouble(value);
}

PyObject* disagreement(PyObject*, PyObject* args) {
  PyObject *indptr_object, *indices_object, *data_object, *z_object;
  if (!PyArg_ParseTuple(args, "OOOO", &indptr_object, &indices_object,
                        &data_object, &z_object)) {
    return nullptr;
  }
  Buffer indptr, indices, data, z;
  if (!indptr.read(indptr_object) || !indices.read(indices_object) ||
      !data.read(data_object) || !z.read(z_object) ||
      !require_indices(indptr, "indptr") ||
      !require_indices(indices, "indices") || !require_double(data, "data") ||
      !require_double(z, "z")) {
    return nullptr;
  }
  const Py_ssize_t n = z.size();
  if (indptr.size() != n + 1 || indices.size() != data.size()) {
    PyErr_SetString(PyExc_ValueError, "incompatible CSR or vector dimensions");
    return nullptr;
  }
  const auto* matrix_data = static_cast<const double*>(data.view.buf);
  const auto* z_data = static_cast<const double*>(z.view.buf);
  double value = 0.0;
  PyThreadState* state = PyEval_SaveThread();
  for (Py_ssize_t row = 0; row < n; ++row) {
    const auto begin = index_at(indptr, row);
    const auto end = index_at(indptr, row + 1);
    for (int64_t p = begin; p < end; ++p) {
      const auto column = index_at(indices, p);
      if (column > row) {
        const double difference = z_data[row] - z_data[column];
        value += matrix_data[p] * difference * difference;
      }
    }
  }
  PyEval_RestoreThread(state);
  return PyFloat_FromDouble(value);
}

PyObject* pd_gradient(PyObject*, PyObject* args) {
  PyObject *indptr_object, *indices_object, *data_object, *z_object, *out_object;
  if (!PyArg_ParseTuple(args, "OOOOO", &indptr_object, &indices_object,
                        &data_object, &z_object, &out_object)) {
    return nullptr;
  }
  Buffer indptr, indices, data, z, out;
  if (!indptr.read(indptr_object) || !indices.read(indices_object) ||
      !data.read(data_object) || !z.read(z_object) || !out.write(out_object) ||
      !require_indices(indptr, "indptr") ||
      !require_indices(indices, "indices") || !require_double(data, "data") ||
      !require_double(z, "z") || !require_double(out, "out")) {
    return nullptr;
  }
  const Py_ssize_t n = z.size();
  if (indptr.size() != n + 1 || indices.size() != data.size() || out.size() != n) {
    PyErr_SetString(PyExc_ValueError, "incompatible CSR or vector dimensions");
    return nullptr;
  }
  const auto* matrix_data = static_cast<const double*>(data.view.buf);
  const auto* z_data = static_cast<const double*>(z.view.buf);
  auto* out_data = static_cast<double*>(out.view.buf);
  PyThreadState* state = PyEval_SaveThread();
  for (Py_ssize_t row = 0; row < n; ++row) {
    double degree = 0.0;
    double product = 0.0;
    const auto begin = index_at(indptr, row);
    const auto end = index_at(indptr, row + 1);
    for (int64_t p = begin; p < end; ++p) {
      const auto column = index_at(indices, p);
      degree += matrix_data[p];
      product += matrix_data[p] * z_data[column];
    }
    out_data[row] = 2.0 * z_data[row] +
                    2.0 * (degree * z_data[row] - product);
  }
  PyEval_RestoreThread(state);
  Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"extremize", extremize, METH_VARARGS, "Compute clipped extremization."},
    {"compute_equilibrium", compute_equilibrium, METH_VARARGS,
     "Compute the fixed-point equilibrium."},
    {"polarization", polarization, METH_VARARGS, "Compute polarization."},
    {"disagreement", disagreement, METH_VARARGS, "Compute disagreement."},
    {"pd_gradient", pd_gradient, METH_VARARGS, "Compute the PD gradient."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_cpp_core",
    "C++ numerical kernels for polar_ai.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__cpp_core() { return PyModule_Create(&module); }
