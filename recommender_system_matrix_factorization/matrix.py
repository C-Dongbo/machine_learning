
class SparseMatrix(object):
    def __init__(self, nrows, ncols):
        self.matrix = [None] * nrows
        self.nrows = nrows
        self.ncols = ncols

    def addRow(self, i, val:dict):
        self.matrix[i] = val

    def get_matrix(self):
        return self.matrix

    def __repr__(self):
        s = ""
        for row in range(min(self.nrows, 10)):
            if row == 0:
                s += "/"
            elif row == self.nrows-1:
                s += "\\"
            else:
                s += "|"
            for col in range(min(self.ncols, 10)):
                if col in self.matrix[row].keys():
                    s += "%6s " % self.matrix[row][col]
                else:
                    s += "%6s " % 0
            if self.ncols > 10:
                s += "... "
            if row == 0:
                s += "\\\n"
            elif row == self.nrows-1:
                s += "/\n"
            else:
                s += "|\n"
        if self.nrows > 10:
            s += "...\n"
        return s
    
    def get_col(self, j):
        col = {}
        for row_idx, col_val in enumerate(self.matrix):
            if j in col_val.keys():
                col[row_idx] = col_val[j]
        return col

    def dot(self, mat):
        result_matrix = SparseMatrix(len(self.matrix),mat.ncols)
        result_row = [0] * len(self.matrix)
        for row_idx, col_val in enumerate(self.matrix):
            col = {}
            for mat_col_idx in range(mat.ncols):
                mat_col_val = mat.get_col(mat_col_idx)
                tmp = 0
                for col_idx in col_val.keys():
                    if col_idx in mat_col_val.keys():
                        tmp += col_val[col_idx] *mat_col_val[col_idx]
                if tmp != 0:
                    col[mat_col_idx] = tmp
            
            result_matrix.addRow(row_idx,col)
        return result_matrix

    def transpose(self):
        result_matrix = SparseMatrix(self.ncols, self.nrows)

        for i in range(self.ncols):
            result_matrix.addRow(i, self.get_col(i))

        return result_matrix





if __name__=='__main__':
    matrix_a = SparseMatrix(2,3)
    matrix_a.addRow(0,{0:3,1:2,2:2})
    matrix_a.addRow(1,{0:1,1:3})
    
    
    matrix_b = SparseMatrix(3,2)
    matrix_b.addRow(0,{0:1})
    matrix_b.addRow(1,{1:1})
    matrix_b.addRow(2,{0:1})
    
    print(matrix_a)
    print(matrix_b)

    print(matrix_a.dot(matrix_b))
