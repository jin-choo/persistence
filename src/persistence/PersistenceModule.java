package persistence;

import java.lang.Math;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Map;

import org.apache.commons.math3.util.CombinatoricsUtils;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.longs.*;
import it.unimi.dsi.fastutil.objects.*;
import org.apache.commons.math3.util.Pair;

public class PersistenceModule {
    protected long timestamp_min = Long.MAX_VALUE;
    protected long timestamp_max = 0;
    protected double time_unit_duration = Double.MAX_VALUE;
    protected int node_max = 0;
    protected LongOpenHashSet[][] set_panel;
    protected ObjectOpenHashSet<Pair<Long, Long>>[][] set_panel2;

    protected int number_time_units;
    protected int size_hoi;

    protected int panel;
    protected int basket_size;
    protected Iterator<int[]> combinationsIterator;
    protected int[] combination;
    protected long hoi_long;
    protected long hoi_long2;

    protected Long2IntLinkedOpenHashMap[] count_panel;
    protected Object2IntLinkedOpenHashMap[] count_panel2;
    protected Long2IntOpenHashMap countpanel;
    protected Object2IntOpenHashMap countpanel2;
    protected LongIterator set_panel_iterator;
    protected ObjectIterator set_panel_iterator2;
    protected ObjectArrayList<Long2IntMap.Entry> countpanel_entries;
    protected ObjectArrayList<Object2IntMap.Entry> countpanel_entries2;

    public PersistenceModule(final int m, final int k) {
        number_time_units = m;
        size_hoi = k;
    }

    public void processTimestamp(final long timestamp) {
        if(timestamp < timestamp_min) timestamp_min = timestamp;
        if(timestamp > timestamp_max) timestamp_max = timestamp;
    }

    public void panelDuration(final int data_coauth) {
        System.out.println("timestamp_min: " + timestamp_min);
        System.out.println("timestamp_max: " + timestamp_max);
        if (data_coauth > 0) {
            number_time_units = (int) (timestamp_max - timestamp_min + 1);
            time_unit_duration = 1;
            System.out.println("number_time_units: " + number_time_units);
        }
        else {
            time_unit_duration = (timestamp_max - timestamp_min) / number_time_units;
            System.out.println("time_unit_duration: " + time_unit_duration);
        }
        set_panel = new LongOpenHashSet[Math.min(size_hoi, 3)][number_time_units];
        if (size_hoi > 3)
            set_panel2 = new ObjectOpenHashSet[size_hoi - 3][number_time_units];
        for (int j = 0; j < number_time_units; j++) {
            for (int i = 0; i < Math.min(size_hoi, 3); i++)
                set_panel[i][j] = new LongOpenHashSet();
            for (int i = 3; i < size_hoi; i++)
                set_panel2[i - 3][j] = new ObjectOpenHashSet<Pair<Long, Long>>();
        }
    }

    public void processNodeMax(final int node_num) {
        if (node_num > node_max) node_max = node_num;
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

    public void processBasket(final IntArrayList basket, final long timestamp, final int k_tuple) {
        panel = (int) Math.min(Math.floor((timestamp - timestamp_min) / time_unit_duration), number_time_units - 1);
        basket_size = basket.size();
        if (basket_size > k_tuple) {
            combinationsIterator = CombinatoricsUtils.combinationsIterator(basket_size, k_tuple + 1);
            while (combinationsIterator.hasNext()) {
                combination = combinationsIterator.next();
                hoi_long = 0;
                for (int i = 0; i < Math.min(k_tuple + 1, 3); i++)
                    hoi_long += pow_function(node_max + 1, Math.min(k_tuple, 2) - i) * Long.valueOf(basket.getInt(combination[i]));
                hoi_long2 = 0;
                for (int i = 3; i < k_tuple + 1; i++)
                    hoi_long2 += pow_function(node_max + 1, k_tuple - i) * Long.valueOf(basket.getInt(combination[i]));
                if (k_tuple < 3)
                    set_panel[k_tuple][panel].add(hoi_long);
                else
                    set_panel2[k_tuple - 3][panel].add(new Pair(hoi_long, hoi_long2));
            }
        }
    }

    public int getNode_max() {
        return node_max;
    }

    public void processCount() {
        count_panel = new Long2IntLinkedOpenHashMap[Math.min(size_hoi, 3)];
        for (int k = 0; k < Math.min(size_hoi, 3); k++) {
            count_panel[k] = new Long2IntLinkedOpenHashMap();
            countpanel = new Long2IntOpenHashMap();
            for (int i = 0; i < number_time_units; i++) {
                set_panel_iterator = set_panel[k][i].iterator();
                while (set_panel_iterator.hasNext())
                    countpanel.addTo(set_panel_iterator.nextLong(), 1);
            }
            countpanel_entries = new ObjectArrayList(countpanel.long2IntEntrySet());
            countpanel_entries.sort(Map.Entry.comparingByKey());
            countpanel_entries.sort(Map.Entry.comparingByValue());
            for (Long2IntMap.Entry countpanel_entry : countpanel_entries)
                count_panel[k].put(countpanel_entry.getLongKey(), countpanel_entry.getIntValue());
        }
        if (size_hoi > 3) {
            count_panel2 = new Object2IntLinkedOpenHashMap[size_hoi - 3];
            for (int k = 3; k < size_hoi; k++) {
                count_panel2[k - 3] = new Object2IntLinkedOpenHashMap();
                countpanel2 = new Object2IntOpenHashMap();
                for (int i = 0; i < number_time_units; i++) {
                    set_panel_iterator2 = set_panel2[k - 3][i].iterator();
                    while (set_panel_iterator2.hasNext())
                        countpanel2.addTo(set_panel_iterator2.next(), 1);
                }
                countpanel_entries2 = new ObjectArrayList(countpanel2.object2IntEntrySet());
                countpanel_entries2.sort(new Comparator<Object2IntMap.Entry>() {
                    @Override
                    public int compare(Object2IntMap.Entry o1, Object2IntMap.Entry o2) {
                        Object2IntMap.Entry<Pair<Long, Long>> o1_ = o1;
                        Object2IntMap.Entry<Pair<Long, Long>> o2_ = o2;
                        if (o1_.getKey().getFirst() > o2_.getKey().getFirst())
                            return 1;
                        else if (o1_.getKey().getFirst() < o2_.getKey().getFirst())
                            return -1;
                        else {
                            if (o1_.getKey().getSecond() > o2_.getKey().getSecond())
                                return 1;
                            else if (o1_.getKey().getSecond() < o2_.getKey().getSecond())
                                return -1;
                            else
                                return 0;
                        }
                    }
                });
                countpanel_entries2.sort(new Comparator<Object2IntMap.Entry>() {
                    @Override
                    public int compare(Object2IntMap.Entry o1, Object2IntMap.Entry o2) {
                        if (o1.getIntValue() > o2.getIntValue())
                            return 1;
                        else if (o1.getIntValue() < o2.getIntValue())
                            return -1;
                        else
                            return 0;
                    }
                });
                for (Object2IntMap.Entry countpanel_entry2 : countpanel_entries2)
                    count_panel2[k - 3].put(countpanel_entry2.getKey(), countpanel_entry2.getIntValue());
            }
        }
    }

    public Long2IntLinkedOpenHashMap[] getCount_panel() {
        return count_panel;
    }

    public Object2IntLinkedOpenHashMap[] getCount_panel2() {
        return count_panel2;
    }

    public Long2IntLinkedOpenHashMap[] getCount_panel_h() {
        return count_panel;
    }

    public Object2IntLinkedOpenHashMap[] getCount_panel_h2() {
        return count_panel2;
    }

    public Long2IntLinkedOpenHashMap[] getCount_panel_h_ex() {
        return count_panel;
    }

    public int getNumber_time_units() {
        return number_time_units;
    }

}
