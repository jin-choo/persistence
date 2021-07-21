package persistence.algorithm;

import persistence.PersistenceModule;

import java.util.Comparator;
import java.util.Map;

import org.apache.commons.math3.util.Pair;

import it.unimi.dsi.fastutil.longs.Long2IntLinkedOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2IntMap;
import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.*;

public class persistence_interval extends PersistenceModule {
    protected int observed_time_units;
    protected long set_panel_element;

    public persistence_interval(final int m, final int k, final int i) {
        super(m, k);
        observed_time_units = i;
    }

    @Override
    public void processCount() {
        count_panel = new Long2IntLinkedOpenHashMap[Math.min(size_hoi, 3)];
        for (int k = 0; k < Math.min(size_hoi, 3); k++) {
            count_panel[k] = new Long2IntLinkedOpenHashMap();
            countpanel = new Long2IntOpenHashMap();
            for (int i = 0; i < number_time_units; i++) {
                set_panel_iterator = set_panel[k][i].iterator();
                while (set_panel_iterator.hasNext()) {
                    set_panel_element = set_panel_iterator.nextLong();
                    if (countpanel.containsKey(set_panel_element)) {
                        if (i < countpanel.get(set_panel_element) % (number_time_units + 1) + observed_time_units)
                            countpanel.addTo(set_panel_element, number_time_units + 1);
                    }
                    else {
                        if (i < number_time_units - observed_time_units + 1)
                            countpanel.put(set_panel_element, number_time_units + 1 + i);
                    }
                }
            }
            countpanel_entries = new ObjectArrayList();
            for (Long2IntMap.Entry countpanel_entry : countpanel.long2IntEntrySet()) {
                countpanel_entry.setValue(countpanel_entry.getIntValue());
                countpanel_entries.add(countpanel_entry);
            }
            countpanel_entries.sort(Map.Entry.comparingByKey());
            countpanel_entries.sort(Map.Entry.comparingByValue());
            for (Long2IntMap.Entry countpanel_entry : countpanel_entries)
                count_panel[k].put(countpanel_entry.getLongKey(), countpanel_entry.getIntValue());
        }
        if (size_hoi > 3) {
            count_panel2 = new Object2IntLinkedOpenHashMap[size_hoi - 3];
            Object set_panel_element2;
            for (int k = 3; k < size_hoi; k++) {
                count_panel2[k - 3] = new Object2IntLinkedOpenHashMap();
                countpanel2 = new Object2IntOpenHashMap();
                for (int i = 0; i < number_time_units; i++) {
                    set_panel_iterator2 = set_panel2[k - 3][i].iterator();
                    while (set_panel_iterator2.hasNext()) {
                        set_panel_element2 = set_panel_iterator2.next();
                        if (countpanel2.containsKey(set_panel_element2)) {
                            if (i < countpanel2.getInt(set_panel_element2) % (number_time_units + 1) + observed_time_units)
                                countpanel2.addTo(set_panel_element2, number_time_units + 1);
                        }
                        else {
                            if (i < number_time_units - observed_time_units + 1)
                                countpanel2.put(set_panel_element2, number_time_units + 1 + i);
                        }
                    }
                }
                countpanel_entries2 = new ObjectArrayList(countpanel2.object2IntEntrySet());
                for (Object2IntMap.Entry<Pair<Long, Long>> countpanel_entry2 : countpanel_entries2)
                    countpanel_entry2.setValue(countpanel_entry2.getIntValue());
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
                for (Object2IntMap.Entry countpanel_entry2_ : countpanel_entries2)
                    count_panel2[k - 3].put(countpanel_entry2_.getKey(), countpanel_entry2_.getIntValue());
            }
        }
    }
}
