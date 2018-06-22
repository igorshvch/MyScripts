#import datetime as dt
import array as ar
import re
import pymorphy2
import sys
import _thread as thread
import tkinter as tk
from tkinter import ttk
import infopovody as infp

VERSION = '0.6.0 dev.ed.'

CREDENTIALS = ['Версия: {}'.format(VERSION),
               'Использованные модули: стандартная библиотека.\n',
               ('ВНИМАНИЕ!\nДанная программа предназначена для работы с'
               +' фрагментами, содержащими в совокупности не более 10000 - 12000'
               +' тысяч слов. Превышение общей длины оцениваемых'
               +' фрагментов может привести к переполнению памяти или вызвать активацию'
               +' антивирусных программ.\n'),
               '\n\n(c) Шевченко И.С.',
               'КонсультантПлюс, ЭАО, 2017г.']

HEADERFONT = ("Verdana", 14)
SIMFONT = ("Verdana", 12, "bold")
INFOFONT = ("Verdana", 11)

parser = pymorphy2.MorphAnalyzer().parse

#tk.Tk().selection_get(selection = "CLIPBOARD")

lock = thread.allocate_lock()   

class CustomTextWidget(tk.Text):

    def __init__(self, *args, **kwargs):
        tk.Text.__init__(self, *args, wrap='word', **kwargs)
        #self.tag_configure('center', justify='left')
        #self.tag_add('center', 1.0, 'end')

        #The following below code was taken from: http://stackoverflow.com/questions/3781670/how-to-highlight-text-in-a-tkinter-text-widget

    def highlight_pattern(self, pattern, tag, start="1.0", end="end",
                          regexp=True):
        #Apply the given tag to all text that matches the given pattern

        #If 'regexp' is set to True, pattern will be treated as a regular
        #expression according to Tcl's regular expression syntax.
        

        start = self.index(start)
        end = self.index(end)
        self.mark_set("matchStart", start)
        self.mark_set("matchEnd", start)
        self.mark_set("searchLimit", end)

        count = tk.IntVar()
        while True:
            index = self.search(pattern, "matchEnd","searchLimit",
                                count=count, regexp=regexp)
            if index == "": break
            if count.get() == 0: break # degenerate pattern which matches zero-length strings
            self.mark_set("matchStart", index)
            self.mark_set("matchEnd", "%s+%sc" % (index, count.get()))
            self.tag_add(tag, "matchStart", "matchEnd")

class MainWin(tk.Tk):#, CustomTextWidget):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        #CustomTextWidget.__init__(self, *args, **kwargs)
        self.entry_txt1 = None
        self.entry_txt2 = None
        self.output_txt1 = None
        self.output_txt2 = None
        #self.text_store = {key:None for key in ['string1', 'string2', 'num_box1', 'num_box2']} #Storing all text data
        self.rb_comp = {}#Rdiobuttons store
        self.rb_morph = {}#Rdiobuttons store
        self.title("Поиск различий в текстах")
        self.values_comparaison = [['Поиск различий', 'diff', lambda:self.mode_shanger('comp')],
                                         ['Поиск общих слов', 'siml', lambda:self.mode_shanger('comp')]]
        self.values_morphology = [['Учитывать морфологию', 'with_morph', lambda:self.mode_shanger('morph')],
                                        ['Не учитывать морфологию', 'without_morph', lambda:self.mode_shanger('morph')]]
        self.tk_rb_var_comparaison = tk.StringVar()#Radiobutton var diff or siml
        self.tk_rb_var_morphology = tk.StringVar()#Radiobutton var with_morph or without_morph
        #frames initiation:start
        f_buffer = tk.Frame(master=self)
        f_radiobuttons_container = ttk.LabelFrame(master=f_buffer)
        f_radiobuttons_label = tk.Frame(master=f_radiobuttons_container)
        f_radiobuttons_box_comparasion = tk.Frame(master=f_radiobuttons_container)
        f_radiobuttons_box_morphology = tk.Frame(master=f_radiobuttons_container)
        f_first_entry_line = tk.Frame(master=f_buffer)
        f_first_entry_line_label = tk.Frame(master=f_first_entry_line)
        f_first_entry_line_text_button = tk.Frame(master=f_first_entry_line)
        f_first_entry_line_text = tk.Frame(master=f_first_entry_line_text_button)
        f_first_entry_line_button = tk.Frame(master=f_first_entry_line_text_button)
        f_second_entry_line = tk.Frame(master=f_buffer)
        f_second_entry_line_label = tk.Frame(master=f_second_entry_line)
        f_second_entry_line_text_button = tk.Frame(master=f_second_entry_line)
        f_second_entry_line_text = tk.Frame(master=f_second_entry_line_text_button)
        f_second_entry_line_button = tk.Frame(master=f_second_entry_line_text_button)
        f_information_box_container = tk.Frame(master=f_buffer)
        f_information_box_container_holder = tk.Frame(master=f_information_box_container)
        f_information_box_container_label = tk.Frame(master=f_information_box_container)
        f_first_output_container = tk.Frame(master=f_buffer)
        f_first_output_ocntainer_label = tk.Frame(master=f_first_output_container)
        f_first_output_container_text = tk.Frame(master=f_first_output_container)
        f_second_output_container = tk.Frame(master=f_buffer)
        f_second_output_container_label = tk.Frame(master=f_second_output_container)
        f_second_output_container_text = tk.Frame(master=f_second_output_container)      
        #frames storing
        self.frames_store = {}
        self.frames = [['buffer', {'f_object':f_buffer,
                                   'f_padx' : 3, 'f_pady' : 3}],
                       ['radiobuttons_container', {'f_object':f_radiobuttons_container, 'f_fill':'x'}], 
                       ['radiobuttons_box_comparasion', {'f_object':f_radiobuttons_box_comparasion, 'f_fill':'x',
                                                         'f_padx' : 0, 'f_pady' : 3}],
                       ['radiobuttons_box_morphology', {'f_object':f_radiobuttons_box_morphology, 'f_fill':'x',
                                                        'f_padx' : 0, 'f_pady' : 3}],
                       ['radiobuttons_label', {'f_object':f_radiobuttons_label, 'f_fill':'x', 'f_pady' : 3}],
                       ['first_entry_line', {'f_object':f_first_entry_line, 'f_fill':'x',
                                             'f_padx' : 3, 'f_pady' : 3}],
                       ['first_entry_line_label', {'f_object':f_first_entry_line_label, 'f_fill':'x'}],
                       ['first_entry_line_text_button', {'f_object':f_first_entry_line_text_button,
                                                         'f_fill':'x'}],
                       ['first_entry_line_text', {'f_object':f_first_entry_line_text,
                                                  'f_side':'left', 'f_fill':'x'}],
                       ['first_entry_line_button', {'f_object':f_first_entry_line_button,
                                                    'f_side':'right', 'f_expand':'no', 'f_fill':'x'}],
                       ['second_entry_line', {'f_object':f_second_entry_line, 'f_fill':'x',
                                              'f_padx' : 3, 'f_pady' : 3}],
                       ['second_entry_line_label', {'f_object':f_second_entry_line_label, 'f_fill':'x'}],
                       ['second_entry_line_text_button', {'f_object':f_second_entry_line_text_button, 'f_fill':'x'}],
                       ['second_entry_line_text', {'f_object':f_second_entry_line_text,
                                                   'f_side':'left', 'f_fill':'x'}],
                       ['second_entry_line_button', {'f_object':f_second_entry_line_button,
                                                     'f_side':'right', 'f_expand':'no', 'f_fill':'x'}],
                       ['information_box_container', {'f_object':f_information_box_container, 'f_fill':'x'}],
                       ['information_box_container_holder', {'f_object':f_information_box_container_holder, 'f_fill':'x',
                                                            'f_side':'top'}],
                       ['information_box_container_label', {'f_object':f_information_box_container_label, 'f_fill':'x',
                                                            'f_side':'top'}],
                       ['first_output_container', {'f_object':f_first_output_container,
                                                   'f_padx' : 3, 'f_pady' : 3}],
                       ['first_output_container_label', {'f_object':f_first_output_ocntainer_label, 'f_fill':'x'}],
                       ['first_output_container_text', {'f_object':f_first_output_container_text}],
                       ['second_output_container', {'f_object':f_second_output_container,
                                                    'f_padx' : 3, 'f_pady' : 3}],
                       ['second_output_container_label', {'f_object':f_second_output_container_label,
                                                          'f_fill':'x'}],
                       ['second_output_container_text', {'f_object':f_second_output_container_text}]]
        #frames initiation:end
        self.options_window = None
        
        self.make_options_window()
        self.make_main_window()

    def new_win(self, title):
        new_window = tk.Toplevel(self)
        new_window.title(title)
        return new_window

    def frame_maker(self, frames):
        frames_dict = {}
        for i in range(len(frames)):
            frames_dict[frames[i][0]] = frames[i][1]['f_object']
            if i != 'radiobuttons_container':
                frames_dict[frames[i][0]].config(relief=frames[i][1].get('f_relief', None),
                                                 bd=frames[i][1].get('f_bd', None))
            frames_dict[frames[i][0]].pack(side=frames[i][1].get('f_side', 'top'),
                                           expand=frames[i][1].get('f_expand', 'yes'),
                                           fill=frames[i][1].get('f_fill', 'both'),
                                           padx=frames[i][1].get('f_padx', None),
                                           pady=frames[i][1].get('f_pady', None))
        return frames_dict

    def make_text_with_scroll(self, widget=None,
                              parent=None, widget_height=None,
                              widget_side='left', widget_expand='yes',
                              widget_fill='both', scroll_='yes',
                              scroll_side='right',
                              scroll_fill='y'):
        if scroll_ == 'yes':
            scroll_bar = tk.Scrollbar(master=parent)
            if widget:
                tk_object = CustomTextWidget
            else:
                tk_object = tk.Text
            if widget_height:
                widget_var = tk_object(master=parent, height=widget_height)
            else:
                widget_var = tk_object(master=parent)        
            scroll_bar.config(command=widget_var.yview)
            widget_var.config(yscrollcommand=scroll_bar.set)
            scroll_bar.pack(side=scroll_side, fill=scroll_fill)
            widget_var.pack(side=widget_side, expand=widget_expand, fill=widget_fill)
            return widget_var
        else:
            if widget:
                tk_object = CustomTextWidget
            else:
                tk_object = tk.Text
            if widget_height:
                widget_var = tk_object(master=parent, height=widget_height)
            else:
                widget_var = tk_object(master=parent)
            widget_var.pack(side=widget_side, expand=widget_expand, fill=widget_fill)
            return widget_var

    def make_radiobuttons(self, parent=None,
                          values=None,
                          pack_side='top',
                          tk_var=None,
                          wdg_state='normal',
                          pack_option='yes'):
        store = {}
        for i in range(len(values)):
            store[values[i][1]]=tk.Radiobutton(master=parent, text=values[i][0],
                                               variable=tk_var,
                                               value=values[i][1],
                                               command=values[i][2],
                                               state=wdg_state)
            if pack_option=='yes':
                store[values[i][1]].pack(side=pack_side, anchor='nw')
            else:
                pass
        tk_var.set(values[0][1])
        return store

    def make_notebook(self, parent, pad_names=[]):
            notebook = ttk.Notebook(master=parent)
            pad_store = {i:tk.Frame(master=notebook) for i in range(len(pad_names))}
            for i in sorted(pad_store.keys()):
                notebook.add(pad_store[i], text=pad_names[i])
            notebook.pack(side='top', expand='yes', fill='both')
            return pad_store

    def custom_insert1(self, event):
        self.entry_txt1.insert(index='1.0', chars=self.selection_get(selection = "CLIPBOARD"))
    def custom_insert2(self, event):
        self.entry_txt2.insert(index='1.0', chars=self.selection_get(selection = "CLIPBOARD"))
    def custom_insert3(self, event):
        self.infp_var.ent.insert(index='0', string=self.selection_get(selection = "CLIPBOARD"))

    def make_options_window(self):
        self.options_window = self.new_win('Настройки')
        self.options_window.withdraw()
        f_mode = tk.LabelFrame(master=self.options_window, text='Выберите режим')
        f_conf = tk.LabelFrame(master=self.options_window, text='Конфигурирование вставки')
        f_mode.pack(padx=1, pady=1)
        f_conf.pack(padx=1, pady=1)
        f_text = tk.Text(master=f_mode)
        f_text.pack(padx=3, pady=3)
        f_text = tk.Text(master=f_conf)
        f_text.pack(padx=3, pady=3)
        

    def make_main_window(self):
        #Menu
        top_menu = tk.Menu(self)
        self.config(menu=top_menu)
        #file = tk.Menu(top, tearoff=False)
        #file.add_command(label='Show all commands',
                         #command=self.show_all_commands,
                         #underline=0)
        #file.add_command(label='Quit', command=self.destroy, underline=0)
        top_menu.add_cascade(label='Настройки', command=lambda:self.options_window.deiconify(), underline=0)
        top_menu.add_cascade(label='О программе', command=lambda:self.show_credentials(), underline=0)
        #Content
        self.frames_store = self.frame_maker(self.frames)
        self.infp_var = infp.GUI(self.frames_store['radiobuttons_container'])
        self.infp_var.root.pack(side='left', padx=1, pady=1)
        self.infp_var.ent.bind('<Control-igrave>', self.custom_insert3)
        self.frames_store['radiobuttons_container'].config(text='Меню')
        self.entry_txt1 = self.make_text_with_scroll(parent=self.frames_store['first_entry_line_text'],
                                                     widget_height=4)
        self.entry_txt1.bind('<Control-igrave>', self.custom_insert1)#ntilde for cyrillic "м" and mcsd platform coding
        self.entry_txt2 = self.make_text_with_scroll(parent=self.frames_store['second_entry_line_text'],
                                                     widget_height=4)
        self.entry_txt2.bind('<Control-igrave>', self.custom_insert2)#ntilde for cyrillic "м" and mcsd platform coding
        self.output_txt1 = self.make_text_with_scroll(widget='custom', parent=self.frames_store['first_output_container_text'],
                                                      widget_height=20)
        self.output_txt1.tag_configure("green", background="#00ff00", overstrike=1)
        self.output_txt2 = self.make_text_with_scroll(widget='custom', parent=self.frames_store['second_output_container_text'],
                                                      widget_height=20)
        self.output_txt2.tag_configure("yellow", background="yellow")
        tk.Button(master=self.frames_store['first_entry_line_button'],
                  text='Сравнить',
                  command=lambda: thread.start_new_thread(self.LCS_main, (self.tk_rb_var_comparaison.get(),)),
                  height=1).pack()
        tk.Button(master=self.frames_store['second_entry_line_button'],
                  text='Очистить',
                  command=lambda: thread.start_new_thread(self.wdg_clean, ()),
                  height=1).pack()
        self.rb_comp = self.make_radiobuttons(parent=self.frames_store['radiobuttons_box_comparasion'],
                                              values=self.values_comparaison,
                                              pack_side='left',
                                              tk_var=self.tk_rb_var_comparaison,
                                              wdg_state='normal')
        self.rb_morph = self.make_radiobuttons(parent=self.frames_store['radiobuttons_box_morphology'],
                                               values=self.values_morphology,
                                               pack_side='left',
                                               tk_var=self.tk_rb_var_morphology,
                                               wdg_state='disabled')
        #Labels
        ##radiobuttons_label
        self.rb_box_label_header = tk.Label(master=self.frames_store['radiobuttons_label'],
                                            text='Значения:')
        self.rb_box_label_header.pack(side='left', expand='no')
        self.rb_box_label_comp = tk.Label(master=self.frames_store['radiobuttons_label'],
                                          text=('сравнение='+self.tk_rb_var_comparaison.get()),
                                          relief='sunken', bd=1, width=14, anchor='w')
        self.rb_box_label_comp.pack(side='left', expand='no', padx=3)
        self.rb_box_label_morph = tk.Label(master=self.frames_store['radiobuttons_label'],
                                           text=('морфология='+self.tk_rb_var_morphology.get()),
                                           relief='sunken', bd=1, width=24, anchor='w')
        self.rb_box_label_morph.pack(side='left', expand='no', padx=3)
        ##information_box_container
        self.idc_label_header = tk.Label(master=self.frames_store['information_box_container_label'],
                                         text='Общих фрагментов:')
        self.idc_label_header.pack(side='left', expand='no')
        self.idc_label_lcs_len = tk.Label(master=self.frames_store['information_box_container_label'],
                                          relief='sunken', bd=1, width=5, anchor='w')
        self.idc_label_lcs_len.pack(side='left', expand='no', padx=3)
        self.idc_label_first_output = tk.Label(master=self.frames_store['information_box_container_label'],
                                               text='Фрагментов в образце №1:')
        self.idc_label_first_output.pack(side='left', expand='no')
        self.idc_label_text1_len = tk.Label(master=self.frames_store['information_box_container_label'],
                                          relief='sunken', bd=1, width=5, anchor='w')
        self.idc_label_text1_len.pack(side='left', expand='no', padx=3)
        self.idc_label_second_output = tk.Label(master=self.frames_store['information_box_container_label'],
                                                text='Фрагментов в образце №2:')
        self.idc_label_second_output.pack(side='left', expand='no')
        self.idc_label_text2_len = tk.Label(master=self.frames_store['information_box_container_label'],
                                          relief='sunken', bd=1, width=5, anchor='w')
        self.idc_label_text2_len.pack(side='left', expand='no', padx=3)
        #Progress
        self.progress = ttk.Progressbar(master=self.frames_store['information_box_container_holder'],
                                           orient='horizontal',
                                           length=200,
                                           mode='indeterminate')
        self.progress.pack(side='left', padx=3, pady=3)

    def show_credentials(self):
        string = '\n'.join(CREDENTIALS)
        new_window = self.new_win('Информация о программе')
        text = self.make_text_with_scroll(parent=new_window, widget_height=None,
                                          widget_side='left', widget_expand='yes',
                                          widget_fill='both', scroll_='yes',
                                          scroll_side='right',
                                          scroll_fill='y')
        text['relief'] = 'flat'
        text['bg'] = '#d4d0c8'
        text.insert('1.0', chars=string)
        text['state'] = 'disabled'

    def wdg_clean(self):
        self.entry_txt1.delete(index1='1.0', index2='end')
        self.entry_txt2.delete(index1='1.0', index2='end')
        self.output_txt1.delete(index1='1.0', index2='end')
        self.output_txt2.delete(index1='1.0', index2='end')
        self.idc_label_lcs_len['text']=''
        self.idc_label_text1_len['text']=''
        self.idc_label_text2_len['text']=''

    def mode_shanger(self, mode):
        def clearer(mode):
            if mode == 'max':
                self.output_txt1.delete(index1='1.0', index2='end')
                self.output_txt2.delete(index1='1.0', index2='end')
                self.idc_label_lcs_len['text']=''
                self.idc_label_text1_len['text']=''
                self.idc_label_text2_len['text']=''
            if mode == 'min':
                self.output_txt1.delete(index1='1.0', index2='end')
                self.idc_label_lcs_len['text']=''
                self.idc_label_text1_len['text']=''
                self.idc_label_text2_len['text']=''
        if mode == 'comp':
            self.output_txt2['state']='normal'
            self.output_txt2['bg']='white'
            if self.tk_rb_var_comparaison.get() == 'diff':
                self.rb_morph['with_morph']['state']='disabled'
                self.rb_morph['without_morph']['state']='disabled'
                self.tk_rb_var_morphology.set('with_morph')
                clearer('max')
                self.rb_box_label_comp['text']='сравнение='+self.tk_rb_var_comparaison.get()
                self.rb_box_label_morph['text']='морфология='+self.tk_rb_var_morphology.get()
            if self.tk_rb_var_comparaison.get() == 'siml':
                self.rb_morph['with_morph']['state']='normal'
                self.rb_morph['without_morph']['state']='normal'
                clearer('max')
                self.output_txt2['state']='disabled'
                self.output_txt2['bg']='#d4d0c8'
                self.rb_box_label_comp['text']='сравнение='+self.tk_rb_var_comparaison.get()
        if mode == 'morph':
            if self.tk_rb_var_morphology.get() == 'with_morph':
                self.rb_box_label_morph['text']='морфология='+self.tk_rb_var_morphology.get()
                clearer('min')
            if self.tk_rb_var_morphology.get() == 'without_morph':
                self.rb_box_label_morph['text']='морфология='+self.tk_rb_var_morphology.get()
                clearer('min')

    def LCS_Length(self, norm_text1, norm_text2):
        m = len(norm_text1)
        n = len(norm_text2)
        c = [ar.array('l', [0 for i in range(n+1)]) for mm in range(m+1)]
        b = [ar.array('l', [0 for i in range(n+1)]) for mm in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if norm_text1[i-1] == norm_text2[j-1]:
                     c[i][j] = c[i-1][j-1] + 1
                     b[i][j] = 9
                else:
                    if c[i-1][j] >= c[i][j-1]:
                        c[i][j] = c[i-1][j]
                        b[i][j] = 7
                    else:
                        c[i][j] = c[i][j-1]
                        b[i][j] = 2
        return c[m][n], b

    def return_LCS_num(self,
                       result_b_table,
                       len_norm_text1, len_norm_text2,
                       num_box1, num_box2):
        
        while (len_norm_text1 != 0) and (len_norm_text2 != 0):
            if result_b_table[len_norm_text1][len_norm_text2] == 9:
                num_box1.append(len_norm_text1-1)
                num_box2.append(len_norm_text2-1)
                len_norm_text1-=1
                len_norm_text2-=1
            elif result_b_table[len_norm_text1][len_norm_text2] == 7:
                len_norm_text1-=1
            else:
                len_norm_text2-=1

    def LCS_main(self, mode):
        with lock:
            
            self.progress.start()
            string1 = self.entry_txt1.get(index1='1.0', index2='end-1c')
            #label info
            self.idc_label_text1_len['text']=len(re.sub("[^\w]", " ", string1).split())
            #
            string1 = string1.replace('\n', ' replace ')
            list1 = re.sub("[^\w]", " ", string1).lower().split()
            string2 = self.entry_txt2.get(index1='1.0', index2='end-1c')
            #label info
            self.idc_label_text2_len['text']=len(re.sub("[^\w]", " ", string2).split())
            #
            string2 = string2.replace('\n', ' replace ')
            list2 = re.sub("[^\w]", " ", string2).lower().split()
            len_list1 = len(list1)
            len_list2 = len(list2)
            if mode == 'diff':
                locker = ['no']
                if len_list1+len_list2 >= 12000:
                    new_window = self.new_win('ВНИМАНИЕ!')
                    tk.Label(master=new_window, text=('Превышена рекомендуемая длина отрывков для сравнения.\n'
                                                      +'Продолжение операции может привести к отказу в работе программы\n'
                                                      +'или активации антивирусных программ')).pack(side='top', expand='no', padx=3, pady=3)
                    tk.Button(master=new_window, text='Прекратить', command=lambda:locker.append('yes')).pack(side='left', expand='no', padx=3, pady=3)
                    tk.Button(master=new_window, text='Продолжить', command=lambda:locker.append('no')).pack(side='left', expand='no', padx=3, pady=3)
                    tk.Button(master=new_window, text='Закрыть', command=new_window.destroy).pack(side='right', expand='no', padx=3, pady=3)
                    new_window.focus_set()
                    new_window.grab_set()
                    new_window.wait_window()
                    if locker[-1] == 'yes':
                        self.progress.stop()
                        return None
                    else: pass
                #if self.tk_rb_var_morphology.get() == 'without_morph':
                #list1 = [parser(word)[0][2] for word in list1]
                #list2 = [parser(word)[0][2] for word in list2]
                len_LCS, result_b_tab = self.LCS_Length(list1, list2)
                num_box_list1 = []
                num_box_list2 = []
                resultlist1 = []
                resultlist2 = []
                #labels info
                self.idc_label_lcs_len['text'] = len_LCS

                self.return_LCS_num(result_b_tab, len_list1, len_list2, num_box_list1, num_box_list2)

                if len_list1 == len_list2 == len_LCS :
                    self.output_txt1.insert(index='1.0', chars='Отрывки идентичны')
                    self.progress.stop()
                else:
                    resultlist1 = list1[:]
                    for i in range(len(resultlist1)):
                        if i not in num_box_list1:
                            resultlist1[i] = resultlist1[i] if resultlist1[i]=='replace' else '*'+resultlist1[i].upper()+'*'        
                    resultlist2 = list2[:]
                    for i in range(len(resultlist2)):
                        if i not in num_box_list2:
                            resultlist2[i] = resultlist2[i] if resultlist2[i]=='replace' else '*'+resultlist2[i].upper()+'*'   
                    result_string1 = ' '.join(resultlist1)
                    result_string1 = ' '*5 + result_string1.replace('replace', ('\n'+' '*4))
                    result_string2 = ' '.join(resultlist2)
                    result_string2 = ' '*5 + result_string2.replace('replace', ('\n'+' '*4))
                    self.output_txt1.insert(index='1.0', chars=result_string1)
                    self.output_txt1.highlight_pattern('\*\w+\*', 'green')
                    self.output_txt2.insert(index='1.0', chars=result_string2)
                    self.output_txt2.highlight_pattern('\*\w+\*', 'yellow')  
                    self.progress.stop()
            if mode == 'siml':
                if self.tk_rb_var_morphology.get() == 'without_morph':
                    list1 = [parser(word)[0][2] for word in list1]
                    list2 = [parser(word)[0][2] for word in list2]
                froz1 = frozenset(list1)
                froz2 = frozenset(list2)
                output = froz1&froz2
                #labels info
                self.idc_label_lcs_len['text'] = len(output)
                result_string = ' '*5 + ' '.join(output).replace('replace', '')
                self.output_txt1.insert(index='1.0', chars=result_string)
                self.progress.stop()
                
            
    
MainWin().mainloop()
