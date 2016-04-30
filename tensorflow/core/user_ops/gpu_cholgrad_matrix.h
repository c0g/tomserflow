namespace tensorflow {
    template <typename T>
    struct Matrix {
        T* dataptr;
        int offset; // offset into parent
        int m; // height of block
        int n; // width of block
        int ld; // row major leading dim of parent
        T* data() { return dataptr + offset; }
        Matrix view(int r0, int r1, int c0, int c1)
        {
            r1 = (r1 == -1) ? m : r1;
            c1 = (c1 == -1) ? n : c1;
            int newm = r1 - r0;
            int newn = c1 - c0;
            int newoffset = r0 * ld + c0;
            return Matrix<T>{ data(), newoffset, newm, newn, ld };
        }
    };
    template <typename T>
    struct L3Par {
        Matrix<T> R;
        Matrix<T> D;
        Matrix<T> B;
        Matrix<T> C;
        L3Par(Matrix<T>& parent, int j, int k);
    };
    template <typename T>
    L3Par<T>::L3Par(Matrix<T>& parent, int j, int k)
    {
        R = parent.view(j, k, 0, j);
        D = parent.view(j, k, j, k);
        B = parent.view(k, -1, 0, j);
        C = parent.view(k, -1, j, k);
    }
}