package persistence;

import java.io.*;

import it.unimi.dsi.fastutil.ints.*;
import it.unimi.dsi.fastutil.longs.Long2IntLinkedOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntLinkedOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import org.apache.commons.math3.util.Pair;

public class Common {
    public static void execute(final PersistenceModule module, final String inputPath, final int size_hoi, final int max_he, final int data_coauth) throws IOException {
        long start = System.currentTimeMillis();

        BufferedReader br_times = new BufferedReader(new FileReader(inputPath + "-times.txt"));
        String line_times = br_times.readLine();
        while (line_times != null) {
            module.processTimestamp(Long.valueOf(line_times));
            line_times = br_times.readLine();
        }
        br_times.close();
        module.panelDuration(data_coauth);
        BufferedReader br_simplices = new BufferedReader(new FileReader(inputPath + "-simplices.txt"));
        String line_simplices = br_simplices.readLine();
        while (line_simplices != null) {
            module.processNodeMax(Integer.valueOf(line_simplices));
            line_simplices = br_simplices.readLine();
        }
        br_simplices.close();
        System.out.println("node_max: " + module.getNode_max());
        System.out.println();

        BufferedReader br_nverts;
        String line_nverts;
        int int_line_nverts;
        IntSortedSet basket;
        br_nverts = new BufferedReader(new FileReader(inputPath + "-nverts.txt"));
        br_simplices = new BufferedReader(new FileReader(inputPath + "-simplices.txt"));
        br_times = new BufferedReader(new FileReader(inputPath + "-times.txt"));
        line_nverts = br_nverts.readLine();
        line_times = br_times.readLine();
        IntArrayList basket_list;
        if (max_he < 25) {
            while (line_nverts != null & line_times != null) {
                int_line_nverts = Integer.valueOf(line_nverts);
                if (int_line_nverts >= 2 & int_line_nverts <= max_he) {
                    basket = new IntRBTreeSet();
                    for (int i = 0; i < int_line_nverts; i++) {
                        line_simplices = br_simplices.readLine();
                        if (line_simplices == null) break;
                        basket.add(Integer.valueOf(line_simplices));
                    }
                    basket_list = new IntArrayList(basket);
                    for (int k = 0; k < size_hoi; k++)
                        module.processBasket(basket_list, Long.valueOf(line_times), k);
                } else {
                    for (int i = 0; i < int_line_nverts; i++) {
                        line_simplices = br_simplices.readLine();
                        if (line_simplices == null) break;
                    }
                }
                line_nverts = br_nverts.readLine();
                line_times = br_times.readLine();
            }
        }
        else  {
            while (line_nverts != null & line_times != null) {
                int_line_nverts = Integer.valueOf(line_nverts);
                if (int_line_nverts >= 2) {
                    basket = new IntRBTreeSet();
                    for (int i = 0; i < int_line_nverts; i++) {
                        line_simplices = br_simplices.readLine();
                        if (line_simplices == null) break;
                        basket.add(Integer.valueOf(line_simplices));
                    }
                    basket_list = new IntArrayList(basket);
                    for (int k = 0; k < size_hoi; k++)
                        module.processBasket(basket_list, Long.valueOf(line_times), k);
                }
                line_nverts = br_nverts.readLine();
                line_times = br_times.readLine();
            }
        }
        br_nverts.close();
        br_simplices.close();
        br_times.close();

        module.processCount();

        long end = System.currentTimeMillis();
        System.out.println("Execution time: " + (end - start) / 1000.0 + "s.");
        System.out.println();
    }

    public static long pow_function(int x, int y) {
        long new_x = new Long(x);
        long new_y = new Long(y);
        long result = 1;
        while (new_y > 0) {
            if ((new_y & 1) == 0) {
                new_x *= new_x;
                new_y >>>= 1;
            } else {
                result *= new_x;
                new_y--;
            }
        }
        return result;
    }

    public static void writeOutputs(final PersistenceModule module, final String outputPath, final String perModeCode, int number_time_units, final int size_hoi, final int max_he, final int data_coauth, final int observed_time_units) throws IOException {
        BufferedWriter bw;
        final int node_max = module.getNode_max();
        long[] node_max_j = {pow_function(node_max + 1, 1), pow_function(node_max + 1, 2)};
        final Long2IntLinkedOpenHashMap[] count_panel = module.getCount_panel();
        Int2IntOpenHashMap counter_panel;
        long k_tuple_long;
        if (data_coauth > 0)
            number_time_units = module.getNumber_time_units();

        for (int i = 0; i < Math.min(size_hoi, 3); i++) {
            if (observed_time_units > 0)
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + max_he + ".txt"));
            else
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + ".txt"));
            counter_panel = new Int2IntOpenHashMap();
            for (Long2IntMap.Entry count_panel_entry : count_panel[i].long2IntEntrySet()) {
                k_tuple_long = count_panel_entry.getLongKey();
                bw.write("(");
                for (int j = i; j > 0; j--) {
                    bw.write(k_tuple_long / node_max_j[j - 1] + ", ");
                    k_tuple_long %= node_max_j[j - 1];
                }
                bw.write(k_tuple_long % (node_max + 1) + "): " + count_panel_entry.getIntValue() + "\n");
                counter_panel.addTo(count_panel_entry.getIntValue(), 1);
            }
            bw.close();

            if (observed_time_units > 0)
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + max_he + "_c.txt"));
            else
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + "_c.txt"));
            for (Int2IntMap.Entry counter_panel_entry : counter_panel.int2IntEntrySet())
                bw.write(counter_panel_entry.getIntKey() + ": " + counter_panel_entry.getIntValue() + "\n");
            bw.close();
        }
        if (size_hoi > 3) {
            final Object2IntLinkedOpenHashMap[] count_panel2 = module.getCount_panel2();
            ObjectIterator<Object2IntMap.Entry<Pair<Long, Long>>> count_panel_entry2;
            Long k_tuple_long2;
            Object2IntMap.Entry<Pair<Long, Long>> count_panel_entry2_next;
            for (int i = 3; i < size_hoi; i++) {
                if (observed_time_units > 0)
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + max_he + ".txt"));
                else
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + ".txt"));
                counter_panel = new Int2IntOpenHashMap();
                count_panel_entry2 = count_panel2[i - 3].object2IntEntrySet().fastIterator();
                while (count_panel_entry2.hasNext()) {
                    count_panel_entry2_next = count_panel_entry2.next();
                    k_tuple_long = count_panel_entry2_next.getKey().getFirst();
                    k_tuple_long2 = count_panel_entry2_next.getKey().getSecond();
                    bw.write("(");
                    for (int j = 2; j > 0; j--) {
                        bw.write(k_tuple_long / node_max_j[j - 1] + ", ");
                        k_tuple_long %= node_max_j[j - 1];
                    }
                    bw.write(k_tuple_long + ", ");
                    for (int j = i - 3; j > 0; j--) {
                        bw.write(k_tuple_long2 / node_max_j[j - 1] + ", ");
                        k_tuple_long2 %= node_max_j[j - 1];
                    }
                    bw.write(k_tuple_long2 + "): " + count_panel_entry2_next.getIntValue() + "\n");
                    counter_panel.addTo(count_panel_entry2_next.getIntValue(), 1);
                }
                bw.close();

                if (observed_time_units > 0)
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + max_he + "_c.txt"));
                else
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + "_c.txt"));
                for (Int2IntMap.Entry counter_panel_entry : counter_panel.int2IntEntrySet())
                    bw.write(counter_panel_entry.getIntKey() + ": " + counter_panel_entry.getIntValue() + "\n");
                bw.close();
            }
        }
    }

    public static void writeOutputs_interval(final PersistenceModule module, final String outputPath, final String perModeCode, int number_time_units, final int size_hoi, final int max_he, final int data_coauth, final int observed_time_units) throws IOException {
        BufferedWriter bw;
        final int node_max = module.getNode_max();
        long[] node_max_j = {pow_function(node_max + 1, 1), pow_function(node_max + 1, 2)};
        final Long2IntLinkedOpenHashMap[] count_panel = module.getCount_panel();
        Int2IntOpenHashMap counter_panel;
        long k_tuple_long;
        if (data_coauth > 0)
            number_time_units = module.getNumber_time_units();
        int count_panel_entry_value;

        for (int i = 0; i < Math.min(size_hoi, 3); i++) {
            if (observed_time_units > 0)
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + max_he + ".txt"));
            else
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + ".txt"));
            counter_panel = new Int2IntOpenHashMap();
            for (Long2IntMap.Entry count_panel_entry : count_panel[i].long2IntEntrySet()) {
                k_tuple_long = count_panel_entry.getLongKey();
                bw.write("(");
                for (int j = i; j > 0; j--) {
                    bw.write(k_tuple_long / node_max_j[j - 1] + ", ");
                    k_tuple_long %= node_max_j[j - 1];
                }
                count_panel_entry_value = count_panel_entry.getIntValue();
                bw.write(k_tuple_long % (node_max + 1) + "): " + count_panel_entry_value / (number_time_units + 1) + " / " + count_panel_entry_value % (number_time_units + 1) + "\n");
                counter_panel.addTo(count_panel_entry_value / (number_time_units + 1), 1);
            }
            bw.close();

            if (observed_time_units > 0)
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + max_he + "_c.txt"));
            else
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + "_c.txt"));
            for (Int2IntMap.Entry counter_panel_entry : counter_panel.int2IntEntrySet())
                bw.write(counter_panel_entry.getIntKey() + ": " + counter_panel_entry.getIntValue() + "\n");
            bw.close();
        }
        if (size_hoi > 3) {
            final Object2IntLinkedOpenHashMap[] count_panel2 = module.getCount_panel2();
            ObjectIterator<Object2IntMap.Entry<Pair<Long, Long>>> count_panel_entry2;
            Long k_tuple_long2;
            Object2IntMap.Entry<Pair<Long, Long>> count_panel_entry2_next;
            for (int i = 3; i < size_hoi; i++) {
                if (observed_time_units > 0)
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + max_he + ".txt"));
                else
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + ".txt"));
                counter_panel = new Int2IntOpenHashMap();
                count_panel_entry2 = count_panel2[i - 3].object2IntEntrySet().fastIterator();
                while (count_panel_entry2.hasNext()) {
                    count_panel_entry2_next = count_panel_entry2.next();
                    k_tuple_long = count_panel_entry2_next.getKey().getFirst();
                    k_tuple_long2 = count_panel_entry2_next.getKey().getSecond();
                    bw.write("(");
                    for (int j = 2; j > 0; j--) {
                        bw.write(k_tuple_long / node_max_j[j - 1] + ", ");
                        k_tuple_long %= node_max_j[j - 1];
                    }
                    bw.write(k_tuple_long + ", ");
                    for (int j = i - 3; j > 0; j--) {
                        bw.write(k_tuple_long2 / node_max_j[j - 1] + ", ");
                        k_tuple_long2 %= node_max_j[j - 1];
                    }
                    count_panel_entry_value = count_panel_entry2_next.getIntValue();
                    bw.write(k_tuple_long2 + "): " + count_panel_entry_value / (number_time_units + 1) + " / " + count_panel_entry_value % (number_time_units + 1) + "\n");
                    counter_panel.addTo(count_panel_entry_value / (number_time_units + 1), 1);
                }
                bw.close();

                if (observed_time_units > 0)
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + max_he + "_c.txt"));
                else
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + "_c.txt"));
                for (Int2IntMap.Entry counter_panel_entry : counter_panel.int2IntEntrySet())
                    bw.write(counter_panel_entry.getIntKey() + ": " + counter_panel_entry.getIntValue() + "\n");
                bw.close();
            }
        }
    }

    public static void writeOutputs_interval_front(final PersistenceModule module, final String outputPath, final String perModeCode, int number_time_units, final int size_hoi, final int max_he, final int data_coauth, final int observed_time_units, final int observed_time_units_features) throws IOException {
        BufferedWriter bw;
        final int node_max = module.getNode_max();
        long[] node_max_j = {pow_function(node_max + 1, 1), pow_function(node_max + 1, 2)};
        final Long2IntLinkedOpenHashMap[] count_panel = module.getCount_panel();
        Int2IntOpenHashMap counter_panel;
        long k_tuple_long;
        if (data_coauth > 0)
            number_time_units = module.getNumber_time_units();
        int count_panel_entry_value;

        for (int i = 0; i < Math.min(size_hoi, 3); i++) {
            if (observed_time_units > 0)
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + observed_time_units_features + "_" + max_he + ".txt"));
            else
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + ".txt"));
            counter_panel = new Int2IntOpenHashMap();
            for (Long2IntMap.Entry count_panel_entry : count_panel[i].long2IntEntrySet()) {
                k_tuple_long = count_panel_entry.getLongKey();
                bw.write("(");
                for (int j = i; j > 0; j--) {
                    bw.write(k_tuple_long / node_max_j[j - 1] + ", ");
                    k_tuple_long %= node_max_j[j - 1];
                }
                count_panel_entry_value = count_panel_entry.getIntValue();
                bw.write(k_tuple_long % (node_max + 1) + "): " + count_panel_entry_value / (number_time_units + 1) + " / " + count_panel_entry_value % (number_time_units + 1) + "\n");
                counter_panel.addTo(count_panel_entry_value / (number_time_units + 1), 1);
            }
            bw.close();

            if (observed_time_units > 0)
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + observed_time_units_features + "_" + max_he + "_c.txt"));
            else
                bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + "_c.txt"));
            for (Int2IntMap.Entry counter_panel_entry : counter_panel.int2IntEntrySet())
                bw.write(counter_panel_entry.getIntKey() + ": " + counter_panel_entry.getIntValue() + "\n");
            bw.close();
        }
        if (size_hoi > 3) {
            final Object2IntLinkedOpenHashMap[] count_panel2 = module.getCount_panel2();
            ObjectIterator<Object2IntMap.Entry<Pair<Long, Long>>> count_panel_entry2;
            Long k_tuple_long2;
            Object2IntMap.Entry<Pair<Long, Long>> count_panel_entry2_next;
            for (int i = 3; i < size_hoi; i++) {
                if (observed_time_units > 0)
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + observed_time_units_features + "_" + max_he + ".txt"));
                else
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + ".txt"));
                counter_panel = new Int2IntOpenHashMap();
                count_panel_entry2 = count_panel2[i - 3].object2IntEntrySet().fastIterator();
                while (count_panel_entry2.hasNext()) {
                    count_panel_entry2_next = count_panel_entry2.next();
                    k_tuple_long = count_panel_entry2_next.getKey().getFirst();
                    k_tuple_long2 = count_panel_entry2_next.getKey().getSecond();
                    bw.write("(");
                    for (int j = 2; j > 0; j--) {
                        bw.write(k_tuple_long / node_max_j[j - 1] + ", ");
                        k_tuple_long %= node_max_j[j - 1];
                    }
                    bw.write(k_tuple_long + ", ");
                    for (int j = i - 3; j > 0; j--) {
                        bw.write(k_tuple_long2 / node_max_j[j - 1] + ", ");
                        k_tuple_long2 %= node_max_j[j - 1];
                    }
                    count_panel_entry_value = count_panel_entry2_next.getIntValue();
                    bw.write(k_tuple_long2 + "): " + count_panel_entry_value / (number_time_units + 1) + " / " + count_panel_entry_value % (number_time_units + 1) + "\n");
                    counter_panel.addTo(count_panel_entry_value / (number_time_units + 1), 1);
                }
                bw.close();

                if (observed_time_units > 0)
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + observed_time_units + "_" + observed_time_units_features + "_" + max_he + "_c.txt"));
                else
                    bw = new BufferedWriter(new FileWriter(outputPath + "_" + perModeCode + (i + 1) + "_" + number_time_units + "_" + max_he + "_c.txt"));
                for (Int2IntMap.Entry counter_panel_entry : counter_panel.int2IntEntrySet())
                    bw.write(counter_panel_entry.getIntKey() + ": " + counter_panel_entry.getIntValue() + "\n");
                bw.close();
            }
        }
    }
}