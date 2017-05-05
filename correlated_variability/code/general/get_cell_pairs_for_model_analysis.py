import random

def get_cell_pairs_for_model_analysis(num_cells, num_pairs):
    cell_pairs = []
    
    while len(cell_pairs) < num_pairs:
        rand_cell1 = random.randint(1, num_cells)
        rand_cell2 = random.randint(1, num_cells)
        if rand_cell1 != rand_cell2:
            if rand_cell1 < 10:
                rand_cell1 = '0%s'%str(rand_cell1)
            if rand_cell2 < 10:
                rand_cell2 = '0%s'%str(rand_cell2)
            pair_name = ['cell_%s'%str(rand_cell1), 'cell_%s'%str(rand_cell2)]
            same_pair = ['cell_%s'%str(rand_cell2), 'cell_%s'%str(rand_cell1)]
            if pair_name not in cell_pairs and same_pair not in cell_pairs:
                cell_pairs.append(pair_name)
    
    return cell_pairs