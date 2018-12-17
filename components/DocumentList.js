import React from 'react';
import { AsyncStorage, Text, TouchableOpacity, FlatList, ScrollView, View, Button, StyleSheet } from 'react-native';

export class DocumentList extends React.Component {

  constructor(props) {
    super(props);
    this.state = {docs: props.docs};
  }

  // componentDidMount() {
  //   this.setState({
  //     docs: this.props.docs,
  //   })
  //   console.log("Mounted new docs, should be visible!")
  // }

  // async componentDidMount() {
  //   let docs = await AsyncStorage.getItem('quizzes') || [];
  //
  //   console.log(docs);
  //   if(docs.length > 0) {
  //     docs = JSON.parse(docs);
  //   }
  //
  //   this.setState({
  //     documents: docs
  //   }, function() {
  //     console.log(this.state.documents);
  //   })
  // }
  //
  // state = {
  //   documents: [],
  // }


  state = {
    docs: [],
  }


  render() {

    console.log("See if any function props gets passed here");
    console.log(this.props.navIntoDoc);
    return (
      <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
        <FlatList
          docs={this.state.docs}
          style={styles.flatList}
          data={this.props.docs}
          keyExtractor={(item, index) => item.docTitle}
          renderItem={({item}) => (
            <TouchableOpacity style={styles.listItem} onPress={() => this.props.navIntoDoc({item})}>
              <Text style={styles.listItemText}>{item.docTitle}</Text>
            </TouchableOpacity>
          )}
        />
      </ScrollView>
    );
  }
}

const styles = StyleSheet.create({
  flatList: {
    backgroundColor: 'yellow',
    flexDirection: 'column',
  },
  listItem: {
    backgroundColor: '#6f3',
    height: 100,
    margin: 10,
    flexDirection: 'column',
    justifyContent: 'center',
    flex: 1,
  },
  listItemText: {
    fontSize: 30,
    textAlign: 'center',
  },
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  contentContainer: {
    flex: 1,
  },
})
